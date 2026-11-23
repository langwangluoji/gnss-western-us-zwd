"""Train frozen-network 100-to-25 km GAN on 2008-2010; validate final model2011."""
from pathlib import Path
import argparse
import hashlib
import json
import random
import time
import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from us_gan import USResidualUNet, PairCritic, sample_points, critic_loss, valid_indices

S=Path(__file__).resolve().parents[1]
DATA=S/'02_data/processed/gan_25km_2008_2011'

def masked_mse(a,b,mask):
    if not mask.any(): return a.sum()*0
    return (a-b).square()[mask].mean()

def load(train_years=(2008,2009,2010),data_path=None,validation_years=(2011,),chunked_normalization=False):
    with h5py.File(DATA/'paired_fields.h5' if data_path is None else data_path,'r') as f:
        data={k:torch.from_numpy(f[k][:]) for k in ['conditions','target_h0','target_mask','station_h0','station_xy','station_height_factor','fine_height_factor','station_sparse']}
        years=f['year'][:]; hours=f['hour_id'][:]
    train=valid_indices(years,train_years); val=valid_indices(years,validation_years)
    if chunked_normalization:
        ids=np.flatnonzero(np.isin(years,train_years))
        total=torch.zeros(15,dtype=torch.float64)
        square=torch.zeros_like(total)
        count=torch.zeros_like(total)
        for offset in range(0,len(ids),512):
            raw=data['conditions'][ids[offset:offset+512]].double()
            mask=torch.ones_like(raw,dtype=torch.bool)
            mask[:,:5]=raw[:,5:6]>.5
            clean=torch.where(mask,raw,0)
            total+=clean.sum((0,2,3))
            square+=clean.square().sum((0,2,3))
            count+=mask.sum((0,2,3))
        mean=total/count.clamp_min(1)
        std=(square/count.clamp_min(1)-mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
        data['mean']=mean.float()[None,None,:,None,None]
        data['std']=std.float()[None,None,:,None,None]
        return data,train,val,years,hours
    raw=data['conditions'][np.flatnonzero(np.isin(years,train_years))].double()
    mask=torch.ones_like(raw,dtype=torch.bool)
    mask[:,:5]=raw[:,5:6]>.5
    count=mask.sum((0,2,3)).clamp_min(1)
    clean=torch.where(mask,raw,0)
    mean=clean.sum((0,2,3))/count
    std=(clean.square().sum((0,2,3))/count-mean.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    data['mean']=mean.float()[None,None,:,None,None]
    data['std']=std.float()[None,None,:,None,None]
    return data,train,val,years,hours

def batch(data,indices,device):
    x=data['conditions'][indices[:,None]-np.array([0,6,12,18])[None,:]].to(device)
    normalized=(x-data['mean'].to(device))/data['std'].to(device)
    normalized[:,:,:5]=torch.where(x[:,:,5:6]>.5,normalized[:,:,:5],0)
    baseline=x[:,0,0:1]/20
    mask=data['target_mask'][indices].to(device)&(F.interpolate(x[:,0,5:6],scale_factor=4,mode='bilinear',align_corners=False)>=1-1e-6)
    return normalized,baseline,mask

def station_arrays(data,result,mask,indices,device):
    xy=data['station_xy'].float().to(device)[None].expand(len(indices),-1,-1)
    inside=(xy.abs()<=1).all(-1)
    xy=xy.clamp(-1,1)
    pred=sample_points(result['field_mm'],xy)
    base=sample_points(result['baseline_mm'],xy)
    labels=data['station_h0'][indices].to(device)/20
    valid=torch.isfinite(labels)&inside&(sample_points(mask.float(),xy)>=1-1e-6)&(~data['station_sparse'].to(device)[None])
    factor=data['station_height_factor'].float().to(device)[None]
    return pred*factor,base*factor,torch.nan_to_num(labels)*factor,valid

def update_critic(critic,optimizer,real,fake,meta):
    if len(real)==0: return fake.sum()*0,0.
    if len(real)>2048:
        take=torch.randperm(len(real),device=real.device)[:2048]
        real,fake,meta=real[take],fake[take],meta[take]
    optimizer.zero_grad(set_to_none=True)
    loss=critic_loss(critic(real.detach(),meta),critic(fake.detach(),meta))
    loss.backward(); optimizer.step()
    for p in critic.parameters(): p.requires_grad_(False)
    g=-critic(fake,meta).mean()
    return g,float(loss.detach())

def validate(model,data,val,years,hours,out,device):
    model.eval()
    totals={k:0. for k in ['sse','baseline_sse','abs','baseline_abs','n']}
    roles=pd.read_csv(S/'02_data/processed/reference_25km_v1/station_roles.csv')
    held=roles[roles.in_main_2008_2010 & roles.role.eq('validation') & roles.in_core].copy()
    # Read spatial validation observations only after fixed-epoch training completes.
    valrows=np.flatnonzero(years==2011)
    truth=np.full((len(hours),len(held)),np.nan,dtype=np.float32)
    for j,row in enumerate(held.itertuples()):
        df=pd.read_parquet(S/f'02_data/processed/pwv_local/period_2007_2011/by_station/{row.station}.parquet',columns=['hour_id','zwd_mm','joint_valid'])
        df=df[df.joint_valid & df.hour_id.between(int(hours[valrows[0]]),int(hours[-1]))]
        truth[df.hour_id.to_numpy()-hours[0],j]=df.zwd_mm.to_numpy()
    vxy=torch.tensor(held[['x_km','y_km']].to_numpy()/300,dtype=torch.float32,device=device)[None]
    vfactor=torch.tensor(np.exp(-held.orthometric_height_m.to_numpy()/3000),dtype=torch.float32,device=device)[None]
    station_error=[[] for _ in range(len(held))]; baseline_error=[[] for _ in range(len(held))]
    factor=data['fine_height_factor'].float().to(device)[None,None]
    station_rows=[]
    with h5py.File(out/'predictions_2011.h5','w') as f,torch.inference_mode():
        f.create_dataset('hour_id',data=hours[val])
        field=f.create_dataset('zwd_h0_mm',shape=(len(val),24,24),dtype='f4',compression='gzip',shuffle=True)
        masks=f.create_dataset('support_mask',shape=(len(val),24,24),dtype='?',compression='gzip')
        for offset in range(0,len(val),128):
            ids=val[offset:offset+128]
            x,base,mask=batch(data,ids,device)
            result=model(x,base)
            pred=result['field_mm']
            label=data['target_h0'][ids].to(device)/20
            e=(pred-label)*factor*20; b=(result['baseline_mm']-label)*factor*20
            totals['sse']+=float(e.square()[mask].sum());totals['baseline_sse']+=float(b.square()[mask].sum())
            totals['abs']+=float(e.abs()[mask].sum());totals['baseline_abs']+=float(b.abs()[mask].sum());totals['n']+=int(mask.sum())
            field[offset:offset+len(ids)]=torch.where(mask,pred*20,float('nan'))[:,0].cpu().numpy()
            masks[offset:offset+len(ids)]=mask[:,0].cpu().numpy()
            grid=vxy.expand(len(ids),-1,-1)
            vp=sample_points(pred,grid)*vfactor*20
            vb=sample_points(result['baseline_mm'],grid)*vfactor*20
            target=torch.tensor(truth[ids],device=device)
            ok=torch.isfinite(target)&(sample_points(mask.float(),grid)>=1-1e-6)
            for j in range(len(held)):
                take=ok[:,j]
                station_error[j].extend((vp[:,j]-target[:,j])[take].cpu().tolist())
                baseline_error[j].extend((vb[:,j]-target[:,j])[take].cpu().tolist())
        f.attrs.update(year=2011,checkpoint='final.pt',height='EGM96 H=0',validation_selection='not used to select checkpoint')
    for j,row in enumerate(held.itertuples()):
        e=np.array(station_error[j]); b=np.array(baseline_error[j])
        if len(e):station_rows.append(dict(station=row.station,n=len(e),model_rmse_mm=float(np.sqrt(np.mean(e*e))),baseline_rmse_mm=float(np.sqrt(np.mean(b*b))),model_bias_mm=float(e.mean())))
    pd.DataFrame(station_rows).to_csv(out/'validation_station_metrics.csv',index=False)
    e=np.array([v for station in station_error for v in station]); b=np.array([v for station in baseline_error for v in station])
    metrics=dict(year=2011,grid_points=int(totals['n']),grid_surface_rmse_mm=(totals['sse']/totals['n'])**.5,baseline_grid_surface_rmse_mm=(totals['baseline_sse']/totals['n'])**.5,grid_surface_mae_mm=totals['abs']/totals['n'],spatial_validation_stations_scored=len(station_rows),station_records=len(e),station_rmse_mm=float(np.sqrt(np.mean(e*e))) if len(e) else None,baseline_station_rmse_mm=float(np.sqrt(np.mean(b*b))) if len(b) else None,independent_test_station_observations_read=False,validation_hours=len(val),notes='Grid accuracy is against interpolated GNSS reference; station scores are against held-out observations; first19 hours of2011 excluded for within-year histories')
    (out/'validation_2011.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('VALIDATION',json.dumps(metrics),flush=True)
    return metrics

def main(args):
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    out=S/'04_models'/args.run_name;out.mkdir(parents=True,exist_ok=True)
    if (out/'final.pt').exists():raise FileExistsError('Use a new run-name; completed model exists')
    data,train,val,years,hours=load()
    model=USResidualUNet(15,16).to(device)
    critics={k:PairCritic().to(device) for k in ['spatial','temporal']}
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    dopts={k:torch.optim.AdamW(v.parameters(),lr=3e-4) for k,v in critics.items()}
    factor=data['fine_height_factor'].float().to(device)[None,None]
    config=dict(seed=42,epochs=args.epochs,warmup_epochs=2,train_samples=len(train),validation_samples=len(val),device=device,train_years=[2008,2009,2010],validation_years=[2011],checkpoint_selection='fixed final epoch',data_manifest_sha256=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((S/'00_protocol/train_25km_2008_2010_validate_2011.json').read_bytes()).hexdigest(),engineering_smoke=bool(args.max_batches))
    (out/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    rng=np.random.default_rng(42);started=time.time();global_step=0
    for epoch in range(args.epochs):
        model.train(); permutation=rng.permutation(train);losses=[];grid_errors=[];nsp=ntp=0
        for offset in range(0,len(permutation),16):
            if args.max_batches and offset>=args.max_batches*16:break
            current=permutation[offset:offset+16]
            ids=np.stack([current,current-1],axis=1).ravel()
            x,base,mask=batch(data,ids,device)
            if not mask.any():continue
            result=model(x,base);pred=result['field_mm'];label=data['target_h0'][ids].to(device)/20
            grid_loss=masked_mse(pred*factor,label*factor,mask)
            sp,sb,sl,sm=station_arrays(data,result,mask,ids,device)
            point_loss=masked_mse(sp,sl,sm)
            loss=grid_loss+.25*point_loss
            if epoch>=2:
                fake=(pred-result['baseline_mm'])*factor
                real=(label-result['baseline_mm'])*factor
                both=mask[:,:,:,1:]&mask[:,:,:,:-1]
                rp=torch.stack([real[:,:,:,1:][both],real[:,:,:,:-1][both]],-1)
                fp=torch.stack([fake[:,:,:,1:][both],fake[:,:,:,:-1][both]],-1)
                meta=pred.new_zeros((len(rp),2));meta[:,0]=.25
                adv,_=update_critic(critics['spatial'],dopts['spatial'],rp,fp,meta)
                loss=loss+.005*adv;nsp+=min(len(rp),2048)
                er=(sl-sb).reshape(-1,2,sl.shape[1]);ef=(sp-sb).reshape(-1,2,sl.shape[1]);ok=sm.reshape(-1,2,sl.shape[1]).all(1)
                rp=er.permute(0,2,1)[ok];fp=ef.permute(0,2,1)[ok]
                meta=pred.new_zeros((len(rp),2));meta[:,0]=1/24
                adv,_=update_critic(critics['temporal'],dopts['temporal'],rp,fp,meta)
                loss=loss+.005*adv;ntp+=min(len(rp),2048)
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            for critic in critics.values():
                for p in critic.parameters():p.requires_grad_(True)
            losses.append(float(loss.detach()));grid_errors.append(float(grid_loss.detach())*400);global_step+=1
        record=dict(epoch=epoch+1,steps=global_step,loss=float(np.mean(losses)),train_grid_rmse_mm=float(np.sqrt(np.mean(grid_errors))),spatial_pairs=nsp,temporal_pairs=ntp,elapsed_seconds=time.time()-started)
        with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        state=dict(model=model.state_dict(),critics={k:v.state_dict() for k,v in critics.items()},optimizer=optimizer.state_dict(),mean=data['mean'],std=data['std'],epoch=epoch+1,config=config,torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state)
        temporary=out/'latest.tmp';torch.save(state,temporary);temporary.replace(out/'latest.pt')
        print(json.dumps(record),flush=True)
    (out/'latest.pt').replace(out/'final.pt')
    if not args.max_batches:validate(model,data,val,years,hours,out,device)
    print('FINISHED',str(out),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=20);p.add_argument('--run-name',default='gan_25km_2008_2010_v1');p.add_argument('--max-batches',type=int,default=0)
    main(p.parse_args())
