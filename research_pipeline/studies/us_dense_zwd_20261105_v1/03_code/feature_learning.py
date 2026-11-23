"""Station-masked SSL and feature/history-gate pilot; selection uses2010 only."""
from pathlib import Path
import json
import time
import hashlib
import numpy as np
import pandas as pd
import h5py
import torch
from torch.nn import functional as F
from us_gan import USResidualUNet, PairCritic, sample_points, valid_indices
from train_us_gan import load, batch, station_arrays, update_critic, masked_mse
from build_reference_grid import weights

S=Path(__file__).resolve().parents[1]
DATA=S/'02_data/processed/gan_25km_2008_2011'
OUT=S/'04_models/feature_learning_pilot_v1'
CACHE=S/'02_data/processed/feature_learning_pilot_v1'
GROUPS={'temperature':[1],'pressure':[2],'Tm':[3],'ERA5_PWV':[4],'terrain':[6],'xy':[7,8],'annual_phase':[9,10],'diurnal_phase':[11,12],'nearest_distance':[13],'neighbor_count':[14]}
LAGS=[6,12,18]

class GatedUNet(USResidualUNet):
    def __init__(self,learn_gates=False,drops=()):
        super().__init__(15,16)
        self.feature_logits=torch.nn.Parameter(torch.zeros(len(GROUPS)),requires_grad=learn_gates)
        self.history_logits=torch.nn.Parameter(torch.zeros(3),requires_grad=learn_gates)
        self.drops=set(drops)

    def gate_input(self,x):
        channel=x.new_ones(15)
        for j,(name,cols) in enumerate(GROUPS.items()):
            channel[cols]=0 if name in self.drops else 2*self.feature_logits[j].sigmoid()
        history=torch.cat([x.new_ones(1),2*self.history_logits.sigmoid()])
        for j,lag in enumerate(LAGS):
            if f'history_{lag}h' in self.drops:history[j+1]=0
        y=x*channel[None,None,:,None,None]*history[None,:,None,None,None]
        # Missingness remains explicit, even when an auxiliary/history branch is dropped.
        y=y.clone();y[:,:,5]=x[:,:,5]
        return y

    def forward(self,x,baseline_mm,reference_mm=None):
        return super().forward(self.gate_input(x),baseline_mm)

    def gate_values(self):
        values={name:float(2*self.feature_logits[j].sigmoid().detach()) for j,name in enumerate(GROUPS)}
        values.update({f'history_{lag}h':float(2*self.history_logits[j].sigmoid().detach()) for j,lag in enumerate(LAGS)})
        return values

    def gate_penalty(self):
        return torch.cat([2*self.feature_logits.sigmoid(),2*self.history_logits.sigmoid()]).mean()

def masked_coarse(xy,observed,query,hidden):
    keep=np.isfinite(observed).all(axis=1)
    keep[np.asarray(hidden,dtype=int)]=False
    ids=np.flatnonzero(keep)
    w,mask,near,count=weights(xy[ids],query,4,2,radius_m=150000,nearest_max_m=100000)
    v=observed[ids].T@w.T
    v[:,~mask]=0
    return np.vstack([v,mask,np.minimum(near/1000,1000),count]).astype('f4')

def prepare_ssl(data,train,hours):
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/'station_masked_ssl.h5'
    indices=train[::6]
    refs=pd.read_csv(DATA/'frozen_network.csv')
    sparse=refs[refs.sparse_input].reset_index(drop=True)
    assert len(sparse)==39 and sparse.role.eq('reference').all()
    xy=sparse[['x_km','y_km']].to_numpy()*1000
    height=sparse.orthometric_height_m.to_numpy()
    if path.exists():
        with h5py.File(path,'r') as f:
            assert np.array_equal(f['indices'][:],indices)
            return {k:torch.from_numpy(f[k][:]) for k in ['conditions','truth_h0','target_mask','hidden_mask','xy','height_factor']}
    observed=np.full((int(train.max())+1,39,5),np.nan,dtype='f4')
    cols=['zwd_mm','temperature_k','pressure_hpa','tm_k','era5_pwv_mm']
    for j,row in enumerate(sparse.itertuples()):
        df=pd.read_parquet(S/f'02_data/processed/pwv_local/period_2007_2011/by_station/{row.station}.parquet',columns=['hour_id','joint_valid',*cols])
        df=df[df.joint_valid & df.hour_id.between(int(hours[0]),int(hours[len(observed)-1]))]
        observed[df.hour_id.to_numpy()-hours[0],j]=df[cols].to_numpy()
    observed[:,:,0]*=np.exp(height[None,:]/3000)
    axis=np.arange(-250000,300000,100000);xx,yy=np.meshgrid(axis,axis);q=np.c_[xx.ravel(),yy.ravel()]
    raw=data['conditions'][indices[:,None]-np.array([0,6,12,18])[None,:]].numpy().copy()
    rng=np.random.default_rng(123)
    hidden_mask=np.zeros((len(indices),39),bool)
    for n,t in enumerate(indices):
        hidden=rng.choice(39,8,replace=False);hidden_mask[n,hidden]=True
        for lag_index,lag in enumerate([0,6,12,18]):
            coarse=masked_coarse(xy,observed[t-lag],q,hidden)
            raw[n,lag_index,:5]=coarse[:5].reshape(5,6,6)
            raw[n,lag_index,5]=coarse[5].reshape(6,6)
            raw[n,lag_index,13:15]=coarse[6:8].reshape(2,6,6)
        if n%500==0:print('SSL cache',n,'/',len(indices),flush=True)
    truth=observed[indices,:,0]
    normalized_xy=torch.tensor(xy/300000,dtype=torch.float32)
    with torch.no_grad():
        support=F.interpolate(torch.from_numpy(raw[:,0,5:6]),scale_factor=4,mode='bilinear',align_corners=False)
        ok=sample_points(support,normalized_xy.clamp(-1,1)[None].expand(len(indices),-1,-1)).numpy()>=1-1e-6
    target_mask=hidden_mask & np.isfinite(truth)&((np.abs(xy)<=300000).all(1)[None])&ok
    result=dict(conditions=raw,truth_h0=np.nan_to_num(truth,nan=0),target_mask=target_mask,hidden_mask=hidden_mask,xy=normalized_xy.numpy(),height_factor=np.exp(-height/3000).astype('f4'))
    with h5py.File(path,'w') as f:
        f.create_dataset('indices',data=indices)
        f.attrs.update(years='2008,2009',hidden_stations_per_sample=8,mask_scope='same station IDs at0/6/12/18hour; all five observation channels before interpolation',observations_role='reference sparse stations only')
        for name,value in result.items():f.create_dataset(name,data=value,compression='gzip',shuffle=True)
    (CACHE/'manifest.json').write_text(json.dumps(dict(samples=len(indices),hidden_station_targets=int(target_mask.sum()),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),years=[2008,2009]),indent=2),encoding='utf-8')
    return {k:torch.from_numpy(v) for k,v in result.items()}

def ssl_pretrain(data,ssl,initial,device):
    torch.manual_seed(42);model=GatedUNet().to(device);model.load_state_dict(initial)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4);rng=np.random.default_rng(42);records=[]
    for epoch in range(8):
        order=rng.permutation(len(ssl['conditions']));errors=[]
        for lo in range(0,len(order),32):
            ids=order[lo:lo+32];mask=ssl['target_mask'][ids].to(device)
            if not mask.any():continue
            raw=ssl['conditions'][ids].to(device)
            x=(raw-data['mean'].to(device))/data['std'].to(device)
            x[:,:,:5]=torch.where(raw[:,:,5:6]>.5,x[:,:,:5],0)
            result=model(x,raw[:,0,0:1]/20)
            xy=ssl['xy'].to(device).clamp(-1,1)[None].expand(len(ids),-1,-1)
            factor=ssl['height_factor'].to(device)[None]
            pred=sample_points(result['field_mm'],xy)*factor
            label=ssl['truth_h0'][ids].to(device)/20*factor
            loss=masked_mse(pred,label,mask)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            errors.append(float(loss.detach())*400)
        row=dict(epoch=epoch+1,hidden_station_rmse_mm=float(np.sqrt(np.mean(errors))))
        records.append(row);print('SSL',json.dumps(row),flush=True)
    pd.DataFrame(records).to_csv(OUT/'ssl_training.csv',index=False)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(state,OUT/'ssl_pretrained.pt')
    return state

def development_truth(hours,years):
    roles=pd.read_csv(S/'02_data/processed/reference_25km_v1/station_roles.csv')
    held=roles[roles.in_main_2008_2010 & roles.role.eq('validation') & roles.in_core].copy()
    truth=np.full((len(hours),len(held)),np.nan,dtype='f4')
    selected=np.flatnonzero(years==2010)
    for j,row in enumerate(held.itertuples()):
        df=pd.read_parquet(S/f'02_data/processed/pwv_local/period_2007_2011/by_station/{row.station}.parquet',columns=['hour_id','zwd_mm','joint_valid'])
        df=df[df.joint_valid & df.hour_id.between(int(hours[selected[0]]),int(hours[selected[-1]]))]
        truth[df.hour_id.to_numpy()-hours[0],j]=df.zwd_mm.to_numpy()
    return held,truth

def evaluate(model,data,ids,held,truth,device):
    model.eval();sse=station_sse=0.;n=station_n=0
    xy=torch.tensor(held[['x_km','y_km']].to_numpy()/300,dtype=torch.float32,device=device)[None]
    sf=torch.tensor(np.exp(-held.orthometric_height_m.to_numpy()/3000),dtype=torch.float32,device=device)[None]
    factor=data['fine_height_factor'].float().to(device)[None,None]
    with torch.inference_mode():
        for lo in range(0,len(ids),128):
            times=ids[lo:lo+128];x,base,mask=batch(data,times,device)
            result=model(x,base);label=data['target_h0'][times].to(device)/20
            error=(result['field_mm']-label)*factor*20
            sse+=float(error.square()[mask].sum());n+=int(mask.sum())
            grid=xy.expand(len(times),-1,-1)
            pred=sample_points(result['field_mm'],grid)*sf*20
            target=torch.tensor(truth[times],device=device)
            ok=torch.isfinite(target)&(sample_points(mask.float(),grid)>=1-1e-6)
            station_sse+=float((pred-target).square()[ok].sum());station_n+=int(ok.sum())
    return dict(grid_rmse_mm=float(np.sqrt(sse/n)),station_rmse_mm=float(np.sqrt(station_sse/station_n)),grid_points=n,station_records=station_n)

def train_arm(name,data,train,initial,learn_gates,device,drops=(),seed=42):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=GatedUNet(learn_gates,drops).to(device);model.load_state_dict(initial)
    critics={k:PairCritic().to(device) for k in ['spatial','temporal']}
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    dopts={k:torch.optim.AdamW(v.parameters(),lr=3e-4) for k,v in critics.items()}
    factor=data['fine_height_factor'].float().to(device)[None,None];records=[];started=time.time()
    for epoch in range(8):
        model.train();order=rng.permutation(train[::6]);errors=[]
        for lo in range(0,len(order),16):
            current=order[lo:lo+16];ids=np.stack([current,current-1],axis=1).ravel()
            x,base,mask=batch(data,ids,device)
            if not mask.any():continue
            result=model(x,base);pred=result['field_mm'];label=data['target_h0'][ids].to(device)/20
            grid_loss=masked_mse(pred*factor,label*factor,mask)
            sp,sb,sl,sm=station_arrays(data,result,mask,ids,device)
            loss=grid_loss+.25*masked_mse(sp,sl,sm)
            if learn_gates:loss=loss+.02*model.gate_penalty()
            if epoch>=2:
                fake=(pred-result['baseline_mm'])*factor;real=(label-result['baseline_mm'])*factor
                both=mask[:,:,:,1:]&mask[:,:,:,:-1]
                rp=torch.stack([real[:,:,:,1:][both],real[:,:,:,:-1][both]],-1)
                fp=torch.stack([fake[:,:,:,1:][both],fake[:,:,:,:-1][both]],-1)
                meta=pred.new_zeros((len(rp),2));meta[:,0]=.25
                adv,_=update_critic(critics['spatial'],dopts['spatial'],rp,fp,meta);loss=loss+.005*adv
                er=(sl-sb).reshape(-1,2,sl.shape[1]);ef=(sp-sb).reshape(-1,2,sl.shape[1]);ok=sm.reshape(-1,2,sl.shape[1]).all(1)
                rp=er.permute(0,2,1)[ok];fp=ef.permute(0,2,1)[ok]
                meta=pred.new_zeros((len(rp),2));meta[:,0]=1/24
                adv,_=update_critic(critics['temporal'],dopts['temporal'],rp,fp,meta);loss=loss+.005*adv
            if not torch.isfinite(loss):raise ValueError('nonfinite pilot loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            for critic in critics.values():
                for p in critic.parameters():p.requires_grad_(True)
            errors.append(float(grid_loss.detach())*400)
        row=dict(arm=name,epoch=epoch+1,train_grid_rmse_mm=float(np.sqrt(np.mean(errors))),seconds=time.time()-started)
        records.append(row);print(json.dumps(row),flush=True)
    pd.DataFrame(records).to_csv(OUT/f'{name}_training.csv',index=False)
    torch.save(dict(model=model.state_dict(),mean=data['mean'],std=data['std'],drops=list(drops),learn_gates=learn_gates,train_years=[2008,2009],epochs=8),OUT/f'{name}.pt')
    return model

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data,train,_,years,hours=load(train_years=(2008,2009))
    dev=valid_indices(years,[2010]);held,truth=development_truth(hours,years)
    torch.manual_seed(42);initial={k:v.clone() for k,v in GatedUNet().state_dict().items()}
    ssl=prepare_ssl(data,train,hours)
    pretrained=ssl_pretrain(data,ssl,initial,device)
    results=[];models={}
    for name,pre,gates in [('baseline',False,False),('SSL_only',True,False),('gate_only',False,True),('SSL_gate',True,True)]:
        model=train_arm(name,data,train,pretrained if pre else initial,gates,device)
        metric=evaluate(model,data,dev,held,truth,device)
        metric.update(arm=name,SSL=pre,gates=gates)
        results.append(metric);models[name]=model
        print('DEVELOPMENT',json.dumps(metric),flush=True)
    table=pd.DataFrame(results);assert table.grid_points.nunique()==table.station_records.nunique()==1
    table.to_csv(OUT/'four_arm_comparison.csv',index=False)
    gate_rows=[]
    for name in ['gate_only','SSL_gate']:
        gate_rows.extend([dict(arm=name,feature=k,gate=v) for k,v in models[name].gate_values().items()])
    pd.DataFrame(gate_rows).to_csv(OUT/'gate_values.csv',index=False)
    # Occlusion is a development diagnostic; hard pruning is separately retrained.
    model=models['SSL_gate'];base=results[3];occlusions=[]
    for feature in list(GROUPS)+[f'history_{lag}h' for lag in LAGS]:
        model.drops={feature}
        metric=evaluate(model,data,dev,held,truth,device)
        metric.update(feature=feature,station_delta_mm=metric['station_rmse_mm']-base['station_rmse_mm'],grid_delta_mm=metric['grid_rmse_mm']-base['grid_rmse_mm'])
        occlusions.append(metric)
    model.drops=set()
    occlusion=pd.DataFrame(occlusions).sort_values(['station_delta_mm','grid_delta_mm'])
    occlusion.to_csv(OUT/'development_occlusion.csv',index=False)
    candidates=occlusion[(occlusion.station_delta_mm<=0)&(occlusion.grid_delta_mm<=0)].feature.head(3).tolist()
    if candidates:
        pruned=train_arm('SSL_pruned',data,train,pretrained,False,device,candidates)
        metric=evaluate(pruned,data,dev,held,truth,device);metric.update(arm='SSL_pruned',SSL=True,gates=False)
        results.append(metric);print('DEVELOPMENT',json.dumps(metric),flush=True)
    pd.DataFrame(results).to_csv(OUT/'all_comparison.csv',index=False)
    best=min(results,key=lambda row:row['station_rmse_mm'])
    result=dict(status='pilot_complete',train_years=[2008,2009],development_year=2010,training_stride_hours=6,epochs=8,seeds=[42],best_arm=best['arm'],candidate_drops=candidates,baseline=results[0],best=best,station_SSL_targets=int(ssl['target_mask'].sum()),observations_2011_used_for_selection=False,independent_test_station_observations_read=False,conclusion_scope='single-seed low-budget development pilot; not global optimal features or definitive2011 improvement')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PILOT COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
