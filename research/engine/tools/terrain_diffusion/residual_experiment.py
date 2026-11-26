"""Budgeted v12 fitting and auditable cached station-level experiments."""
import argparse,copy,csv,json,time
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from .data import Dataset,PROJECT,sha,write_json
from .metrics import evaluate
from .model import v_target
from .residual_field import Foundation,GaussianResidual,ResidualDenoiser,spatial_folds,features_from_batch,spatial_context_query,flat_features,masked_mse

ROOT=PROJECT/'results/20261124_residual_field_v12'


def load_pack(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}


def save_pack(path,items):
    np.savez_compressed(path,**{k:np.stack([v[k] for v in items]) for k in items[0]})


def pack_item(d,data,row,query):
    static,history=features_from_batch(d,data.mean[0],data.std[0])
    n=len(query);size=128;values=dict(static=static,history=history,geo=d['qgeo'][0].numpy(),y=d['target'][0].numpy(),mask=d['mask'][0].numpy(),query_valid=np.ones(n,dtype=bool))
    if n<size:
        for k,v in values.items():values[k]=np.pad(v,[(0,size-n)]+[(0,0)]*(v.ndim-1))
    values['row']=np.array(row);return values


def prepare(root,small=False):
    cache=root/'cache';cache.mkdir(parents=True,exist_ok=True)
    if (cache/'ready.json').exists():return
    data=Dataset(device='cpu');fold=spatial_folds(data.a['geo'],data.a['groups']);np.save(cache/'folds.npy',fold)
    rng=np.random.default_rng(12024);n=24 if small else 512;noof=24 if small else 128
    for held in (-1,0,1,2):
        pool=np.flatnonzero(fold!=held);items=[]
        for i in range(n):
            row=i%72;t=int(rng.integers(12,36));c,q=spatial_context_query(data,rng,row,t,pool,i%2==0)
            items.append(pack_item(data.batch(row,t,c,q),data,row,q))
        save_pack(cache/f'train_{held}.npz',items)
        if held>=0:
            items=[]
            for i in range(noof):
                row=i%72;t=int(rng.integers(12,36));c,q=spatial_context_query(data,rng,row,t,pool,False,np.flatnonzero(fold==held))
                assert not np.any(fold[c]==held) and np.all(fold[q]==held)
                items.append(pack_item(data.batch(row,t,c,q),data,row,q))
            save_pack(cache/f'oof_{held}.npz',items)
        print(json.dumps(dict(stage='cache',fold=held)),flush=True)
    items=[]
    for li,c in enumerate(data.layouts()):
        for row in range(12):
            for hour in range(24):
                q=np.arange(137);d=data.batch(row,12+hour,c,q,'validation');v=pack_item(d,data,row,q);v['layout']=np.array(li);v['hour']=np.array(hour);items.append(v)
    save_pack(cache/'validation.npz',items)
    write_json(cache/'ready.json',dict(data_sha256=data.contract['data_sha256'],fold_counts=np.bincount(fold).tolist(),small=small,
        description='three150km spatial folds; held-out fold excluded from foundation fitting AND conditioning; row modulo6==5 reserved temporal monitor/calibration',
        packs={p.name:sha(p) for p in cache.glob('*.npz')}))


def flat_numpy(pack):
    h=pack['history'];return np.concatenate([pack['static'],h[:,:,-1],h.mean(-2),h[:,:,-1]-h[:,:,0]],-1).astype('f4')


def tensor_pack(pack):
    return {k:torch.as_tensor(v,device='cuda') for k,v in pack.items() if k in ('static','history','geo','y','mask','features','residual')}


def torch_predict(model,pack,chunk=16):
    out=[];model.eval()
    with torch.no_grad():
        for i in range(0,len(pack['y']),chunk):
            s=torch.as_tensor(pack['static'][i:i+chunk],device='cuda');h=torch.as_tensor(pack['history'][i:i+chunk],device='cuda')
            out.append(model(s,h).cpu().numpy())
    return np.concatenate(out)


class Deadline(xgb.callback.TrainingCallback):
    def __init__(self,seconds):self.end=time.monotonic()+seconds
    def after_iteration(self,model,epoch,evals_log):return time.monotonic()>=self.end


def fit_tree(x,y,mask,monitor,folder,seconds,seed):
    folder.mkdir(parents=True,exist_ok=True);started=time.monotonic();fit=mask&~monitor[:,None];check=mask&monitor[:,None]
    if not fit.any() or not check.any():raise ValueError('empty fit or monitor split')
    a=xgb.DMatrix(x[fit],label=y[fit]);b=xgb.DMatrix(x[check],label=y[check]);history={}
    model=xgb.train(dict(objective='reg:squarederror',max_depth=5,eta=.035,subsample=.8,colsample_bytree=.85,min_child_weight=20,reg_lambda=10,seed=seed,nthread=4,tree_method='hist'),a,
        num_boost_round=10000,evals=[(a,'train'),(b,'monitor')],early_stopping_rounds=100,evals_result=history,callbacks=[Deadline(seconds)],verbose_eval=False)
    model=model[:model.best_iteration+1];model.save_model(folder/'tree.ubj');write_json(folder/'tree_history.json',history)
    write_json(folder/'fit.json',dict(seconds=time.monotonic()-started,rounds=model.num_boosted_rounds(),seed=seed,budget_seconds=seconds));return model


def fit_foundation(kind,pack,folder,seconds,seed):
    folder.mkdir(parents=True,exist_ok=True);monitor=pack['row']%6==5
    if kind=='xgboost':return fit_tree(flat_numpy(pack),pack['y'],pack['mask'],monitor,folder,seconds,seed)
    torch.manual_seed(seed);np.random.seed(seed);model=Foundation(kind).cuda();ema=copy.deepcopy(model).eval();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    p=tensor_pack(pack);train=np.flatnonzero(~monitor);val=np.flatnonzero(monitor);rng=np.random.default_rng(seed);start=time.monotonic();step=0;best=float('inf');rows=[]
    while time.monotonic()-start<seconds or step==0:
        step+=1;model.train();idx=rng.choice(train,min(4,len(train)),False);opt.zero_grad(set_to_none=True)
        pred=model(p['static'][idx],p['history'][idx]);loss=masked_mse(pred,p['y'][idx],p['mask'][idx]);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(loss) or not torch.isfinite(gn):raise FloatingPointError('foundation nonfinite')
        opt.step()
        with torch.no_grad():
            for e,v in zip(ema.parameters(),model.parameters()):e.lerp_(v,.01)
        if step%200==0 or step==1:
            with torch.no_grad():
                pred=ema(p['static'][val],p['history'][val]);score=float(masked_mse(pred,p['y'][val],p['mask'][val]).sqrt())
            rows.append(dict(step=step,seconds=time.monotonic()-start,loss=float(loss.detach()),monitor_rmse_normalized=score,gradient_norm=float(gn)))
            if score<best:best=score;torch.save(ema.state_dict(),folder/'best.pt')
            write_json(folder/'progress.json',rows[-1])
            torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),step=step),folder/'last.pt')
            if step%2000==0:print(json.dumps(dict(stage=folder.as_posix(),**rows[-1])),flush=True)
    model.load_state_dict(torch.load(folder/'best.pt',weights_only=True));model.eval()
    with (folder/'history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    write_json(folder/'fit.json',dict(seconds=time.monotonic()-start,steps=step,best_monitor_rmse_normalized=best,seed=seed,budget_seconds=seconds));return model


def predict_foundation(model,kind,pack):
    if kind=='xgboost':
        x=flat_numpy(pack);return model.predict(xgb.DMatrix(x.reshape(-1,113))).reshape(pack['y'].shape)
    return torch_predict(model,pack)


def load_foundation(folder,kind):
    if kind=='xgboost':model=xgb.Booster();model.load_model(folder/'tree.ubj');return model
    model=Foundation(kind).cuda();model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True));return model.eval()


def selection_indices(pack):return np.flatnonzero(np.isin(pack['row'],[0,3,6,9])&(pack['hour']<4))


def score_samples(samples,pack,data):
    truth=np.where(pack['mask'],pack['y']*data.std[0]+data.mean[0],np.nan);scores=[]
    for li in range(3):
        ix=np.flatnonzero(pack['layout']==li);s=samples[:,ix];y=truth[ix];r=evaluate(s,y)
        differences=[];dy=[]
        for month in np.unique(pack['row'][ix]):
            j=np.flatnonzero(pack['row'][ix]==month)
            if len(j)>1:differences.append(np.diff(s[:,j],axis=1));dy.append(np.diff(y[j],axis=0))
        r['increment']=evaluate(np.concatenate(differences,1),np.concatenate(dy)) if differences else {};scores.append(dict(layout=li,**r))
    return dict(macro_rmse_mm=float(np.mean([v['station_macro_rmse_mm'] for v in scores])),crps_mm=float(np.mean([v['fair_crps_mm'] for v in scores])),layouts=scores)


def subset(pack,indices):return {k:v[indices] for k,v in pack.items()}


def save_evaluation(folder,samples,pack,data):
    folder.mkdir(parents=True,exist_ok=True);truth=np.where(pack['mask'],pack['y']*data.std[0]+data.mean[0],np.nan)
    np.savez_compressed(folder/'evaluation.npz',samples=samples,truth=truth,layout=pack['layout'],row=pack['row'],hour=pack['hour'])
    write_json(folder/'metrics.json',score_samples(samples,pack,data))


def foundation_run(root,eid,kind,budget,small):
    data=Dataset(device='cpu');val=load_pack(root/'cache/validation.npz');out=root/eid;out.mkdir(exist_ok=True)
    for seed in (42,17):
        sd=out/f's{seed}';sd.mkdir(exist_ok=True);oof=[]
        for held in (0,1,2):
            p=load_pack(root/f'cache/train_{held}.npz');model=fit_foundation(kind,p,sd/f'fold{held}',budget/8,seed+held*100)
            o=load_pack(root/f'cache/oof_{held}.npz');o['base']=predict_foundation(model,kind,o);oof.append(o);del model
        pack={k:np.concatenate([p[k] for p in oof]) for k in oof[0]};np.savez_compressed(sd/'oof.npz',**pack)
        model=fit_foundation(kind,load_pack(root/'cache/train_-1.npz'),sd/'final',budget/8,seed)
        base=predict_foundation(model,kind,val);np.save(sd/'validation_base.npy',base);samples=(base*data.std[0]+data.mean[0])[None]
        ix=selection_indices(val);save_evaluation(sd/'selection',samples[:,ix],subset(val,ix),data);save_evaluation(sd/'expanded',samples,val,data)
        write_json(sd/'completion.json',dict(status='completed',kind=kind,seed=seed));del model;torch.cuda.empty_cache()
    write_json(out/'completion.json',dict(status='completed',kind=kind,budget_seconds=budget))


def choose_base(root):
    values=[]
    for eid,kind in [('E1201_mlp','mlp'),('E1202_cnn_lstm','cnn_lstm'),('E1203_xgboost','xgboost')]:
        scores=[json.loads((root/eid/f's{s}/selection/metrics.json').read_text())['macro_rmse_mm'] for s in (42,17)]
        values.append(dict(experiment=eid,kind=kind,score=float(np.mean(scores))))
    selected=min(values,key=lambda x:x['score']);write_json(root/'foundation_selection.json',dict(selected=selected,candidates=values,rule='mean two-seed quarterly RMSE; not expanded calendar'));return selected


def downstream_features(pack,base):return np.concatenate([flat_numpy(pack),base[...,None]],-1).astype('f4')


def gaussian_noise(geo,members,generator):
    d=torch.cdist(geo[...,:2],geo[...,:2]);cov=.7*torch.exp(-.5*(d/50)**2)+.2*torch.exp(-.5*(d/150)**2)+.1*torch.eye(geo.shape[-2],device=geo.device)
    chol=torch.linalg.cholesky(cov);z=torch.randn((members,geo.shape[-2]),device=geo.device,generator=generator);return z@chol.T


def probability_predict(model,arm,features,geo,base,scale,data,members=16,seed=12077,query_valid=None):
    gen=torch.Generator(device='cuda').manual_seed(seed);pred=[];model.eval()
    with torch.no_grad():
        for i in range(len(base)):
            valid=np.ones(base.shape[1],dtype=bool) if query_valid is None else query_valid[i]
            f=torch.as_tensor(features[i,valid],device='cuda')[None];g=torch.as_tensor(geo[i,valid],device='cuda')
            if arm=='gaussian':
                mu,sd=model(f);r=mu+sd*gaussian_noise(g,members,gen)
            else:r=model.sample(f.expand(members,-1,-1),g[None].expand(members,-1,-1),torch.randn((members,len(g)),device='cuda',generator=gen))
            full=np.zeros((members,base.shape[1]),dtype='f4');full[:,valid]=(base[i,valid][None]+scale*r.cpu().numpy())*data.std[0]+data.mean[0];pred.append(full)
            if len(base)>100 and (i+1)%144==0:print(json.dumps(dict(stage='inference',arm=arm,done=i+1,total=len(base))),flush=True)
    return np.stack(pred,1)


def fit_probability(root,eid,arm,budget,small):
    data=Dataset(device='cpu');choice=json.loads((root/'foundation_selection.json').read_text())['selected'];val=load_pack(root/'cache/validation.npz');ix=selection_indices(val);vp=subset(val,ix)
    out=root/eid;out.mkdir(exist_ok=True)
    for seed in (42,17):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);sd=out/f's{seed}';sd.mkdir(exist_ok=True)
        pack=load_pack(root/choice['experiment']/f's{seed}/oof.npz');base=np.load(root/choice['experiment']/f's{seed}/validation_base.npy')
        monitor=pack['row']%6==5;train=np.flatnonzero(~monitor);residual=pack['y']-pack['base'];scale=float(np.std(residual[pack['mask']&~monitor[:,None]]));scale=max(scale,.05)
        features=downstream_features(pack,pack['base']);vf=downstream_features(val,base);model=(GaussianResidual() if arm=='gaussian' else ResidualDenoiser()).cuda();ema=copy.deepcopy(model).eval()
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);f=torch.tensor(features,device='cuda');g=torch.tensor(pack['geo'],device='cuda');r=torch.tensor(residual/scale,device='cuda');mask=torch.tensor(pack['mask'],device='cuda');rows=[];best=float('inf');start=time.monotonic();step=0;last_eval=-1e9
        # Budget counts fitting updates; validation cost is reported separately.
        spent=0.;eval_seconds=0.
        while spent<budget/2 or step==0:
            tick=time.monotonic();step+=1;model.train();j=int(rng.choice(train));active=mask[j];fj=f[j:j+1,active];gj=g[j:j+1,active];target=r[j:j+1,active];opt.zero_grad(set_to_none=True)
            if arm=='gaussian':
                mu,sigma=model(fj);d=torch.cdist(gj[0,:,:2],gj[0,:,:2]);k=.7*torch.exp(-.5*(d/50)**2)+.2*torch.exp(-.5*(d/150)**2)+.1*torch.eye(len(d),device='cuda');ch=torch.linalg.cholesky(k)
                z=((target-mu)/sigma)[0];white=torch.linalg.solve_triangular(ch,z[:,None],upper=False);loss=.5*white.square().mean()+sigma.log().mean()+ch.diag().log().mean()
            else:
                t=torch.randint(1,101,(1,),device='cuda');ab=model.ab[t][:,None];noise=torch.randn_like(target);x=ab.sqrt()*target+(1-ab).sqrt()*noise
                v=model(x,t,fj,gj);loss=(v-v_target(target,noise,ab)).square().mean()
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            if not torch.isfinite(loss) or not torch.isfinite(gn):raise FloatingPointError('residual nonfinite')
            opt.step()
            with torch.no_grad():
                for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,.01)
            spent+=time.monotonic()-tick
            if spent-last_eval>=max(60,budget/12) or last_eval<0 or spent>=budget/2:
                check=time.monotonic();samples=probability_predict(ema,arm,vf[ix],val['geo'][ix],base[ix],scale,data,16,12077)
                score=score_samples(samples,vp,data);rows.append(dict(step=step,train_seconds=spent,loss=float(loss.detach()),gradient_norm=float(gn),validation_rmse_mm=score['macro_rmse_mm'],validation_crps_mm=score['crps_mm']))
                if score['crps_mm']<best:best=score['crps_mm'];torch.save(ema.state_dict(),sd/'best.pt');save_evaluation(sd/'selection',samples,vp,data)
                write_json(sd/'progress.json',rows[-1]);torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),step=step),sd/'last.pt')
                print(json.dumps(dict(experiment=eid,seed=seed,**rows[-1])),flush=True);last_eval=spent;eval_seconds+=time.monotonic()-check
        model.load_state_dict(torch.load(sd/'best.pt',weights_only=True));model.eval()
        np.savez_compressed(sd/'normalization.npz',scale=np.array(scale))
        with (sd/'history.csv').open('w',newline='') as file:
            w=csv.DictWriter(file,rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
        samples=probability_predict(model,arm,vf,val['geo'],base,scale,data,16,12099);save_evaluation(sd/'expanded',samples,val,data)
        if arm=='diffusion':
            cal=np.flatnonzero(monitor);calpack=subset(pack,cal);cs=probability_predict(model,arm,features[cal],pack['geo'][cal],pack['base'][cal],scale,data,16,12111,query_valid=pack['query_valid'][cal])
            np.savez_compressed(sd/'calibration.npz',features=features[cal],samples=cs,y=pack['y'][cal],mask=pack['mask'][cal],row=pack['row'][cal])
        write_json(sd/'completion.json',dict(status='completed',steps=step,training_seconds=spent,selection_seconds=eval_seconds,total_seconds=time.monotonic()-start,residual_scale=scale,seed=seed))
        del model,ema;torch.cuda.empty_cache()
    write_json(out/'completion.json',dict(status='completed',arm=arm,budget_seconds=budget))


def tree_correction(root,eid,post,budget,small):
    data=Dataset(device='cpu');choice=json.loads((root/'foundation_selection.json').read_text())['selected'];val=load_pack(root/'cache/validation.npz');out=root/eid;out.mkdir(exist_ok=True)
    for seed in (42,17):
        sd=out/f's{seed}';sd.mkdir(exist_ok=True);base=np.load(root/choice['experiment']/f's{seed}/validation_base.npy')
        if post:
            p=load_pack(root/f'E1206_diffusion/s{seed}/calibration.npz');mean=(p['samples'].mean(0)-data.mean[0])/data.std[0];spread=p['samples'].std(0)/data.std[0]
            x=np.concatenate([p['features'],mean[...,None],spread[...,None]],-1);target=p['y']-mean
            monitor=(p['row']//6)%3==2
        else:
            p=load_pack(root/choice['experiment']/f's{seed}/oof.npz');x=downstream_features(p,p['base']);target=p['y']-p['base'];monitor=p['row']%6==5
        model=fit_tree(x,target,p['mask'],monitor,sd,budget/2,seed)
        for scope in ('selection','expanded'):
            ix=selection_indices(val) if scope=='selection' else np.arange(len(val['y']));v=subset(val,ix);vf=downstream_features(v,base[ix])
            if post:
                old=load_pack(root/f'E1206_diffusion/s{seed}/{scope}/evaluation.npz');samples=old['samples'];mean=(samples.mean(0)-data.mean[0])/data.std[0];spread=samples.std(0)/data.std[0];vf=np.concatenate([vf,mean[...,None],spread[...,None]],-1)
            else:samples=(base[ix]*data.std[0]+data.mean[0])[None]
            correction=model.predict(xgb.DMatrix(vf.reshape(-1,vf.shape[-1]))).reshape(base[ix].shape)*data.std[0]
            save_evaluation(sd/scope,samples+correction[None],v,data)
        write_json(sd/'completion.json',dict(status='completed',seed=seed,post=post))
    write_json(out/'completion.json',dict(status='completed',budget_seconds=budget))
