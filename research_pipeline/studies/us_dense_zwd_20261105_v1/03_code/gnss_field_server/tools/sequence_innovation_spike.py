"""Frozen generative sequence + input-A observation innovations, B-only ridge."""
import json
import time
import numpy as np
import torch
from scipy.spatial.distance import cdist
import explore_sequence_mechanisms as exp
import sequence_mechanism_spike as core
from sequence_grid_spike import XY,TERRAIN,QI,QW,GridMean,GridRandom
from gnss_field.artifacts import write_json,sha256

ROOT=core.PROJECT/'results/20261123_sequence_innovation_v04'
SOURCE=exp.ROOT


def correction_features(a,name,mu):
    raw=a[name+'_raw'];valid=np.isfinite(raw[...,0])
    error=np.where(valid,raw[...,0]-mu[...,a['aids']],0.)
    nodes,inverse=np.unique(QI,return_inverse=True);index=inverse.reshape(-1,4)
    dist=cdist(XY[nodes],a['xy'][a['aids']])/1000
    dz=TERRAIN[nodes,0,None]*3000-a['heights'][a['aids']][None]
    fields=[]
    for radius in (50,150,300):
        for height in (False,True):
            w=np.exp(-.5*(dist/radius)**2)*(dist<=3*radius)
            if height:w*=np.exp(-.5*(dz/750)**2)
            den=np.einsum('qa,bta->btq',w,valid,optimize=True)
            num=np.einsum('qa,bta->btq',w,error,optimize=True)
            f=np.divide(num,den,out=np.zeros_like(num),where=den>1e-8)
            fields.append((f[:,:,index]*QW).sum(-1))
    feat=np.stack(fields,-1)
    return np.concatenate([feat[:,6-lag:30-lag] for lag in (0,1,3)],-1).astype('f4')


def design(feat,height,ids,columns,scales):
    n=len(ids);factor=height[ids]
    intercept=np.broadcast_to(factor,(*feat.shape[:2],n))[...,None]
    if columns==0:return intercept
    return np.concatenate([intercept,feat[:,:,ids,:columns]*factor[None,None,:,None]/scales[:columns]],-1)


def point_score(p,y,m):
    n=p.shape[-1];out=core.metrics(p.reshape(-1,n),y.reshape(-1,n),m.reshape(-1,n))
    dm=m[:,1:]&m[:,:-1]
    out['increment_rmse']=float(np.sqrt((((np.diff(p,axis=1)-np.diff(y,axis=1))**2)*dm).sum()/dm.sum()))
    return out


def probability_score(samples,y,m):
    s=torch.tensor(samples);yt=torch.tensor(y);mt=torch.tensor(m)
    low,high=np.quantile(samples,[.05,.95],axis=1)
    return dict(crps=float(core.fair_crps(s,yt,mt)),increment_crps=float(core.fair_crps(s[:,:,1:]-s[:,:,:-1],yt[:,1:]-yt[:,:-1],mt[:,1:]&mt[:,:-1])),
        coverage90=float((((y>=low)&(y<=high))&m).sum()/m.sum()),width90=float(((high-low)*m).sum()/m.sum()))


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(2);started=time.monotonic()
    a=dict(np.load(SOURCE/'data.npz'));source=json.loads((SOURCE/'results.json').read_text())
    assert sha256(SOURCE/'data.npz')==source['runtime']['data_sha256']
    hours=np.array([int(np.datetime64(f'2018-{m:02d}-27','h').astype('i8'))+np.arange(-6,24) for m in range(1,13)])
    store=core.PackedStore(core.DATA,purpose='train')
    a['reserved_hours']=hours;a['reserved_raw']=store.context_values(hours.ravel())[...,:3].reshape(12,30,len(a['aids']),3)
    a['reserved_y']=np.full((12,30,len(a['heights'])),np.nan,dtype='f4')
    a['reserved_y'][...,a['bids']]=store.targets('B',hours.ravel()).reshape(12,30,len(a['bids']))
    for name in ('train','inner','outer','confirmation'):assert not np.intersect1d(hours,a[name+'_hours']).size
    write_json(ROOT/'experiment.lock.json',dict(code_sha256=sha256(__file__),protocol_sha256=sha256(ROOT/'PROTOCOL.md'),
        source_data=source['runtime']['data_sha256'],source_models=source['runtime']['checkpoints'],
        reserved_hours=hours.tolist(),C_D_labels_read=False,adaptive_after_v03=True,
        dependency_sha256={n:sha256(core.PROJECT/'tools'/n) for n in ['explore_sequence_mechanisms.py','sequence_grid_spike.py','sequence_mechanism_spike.py']}))
    fit=a['split_fit'];inner=a['split_inner'];outer=a['split_outer'];height=np.exp(-a['heights']/3000)
    cache={};models={};selections={}
    for seed in (42,17):
        model=GridMean(27,False).cuda()
        ckpt=SOURCE/f'distance_{seed}_2000.pt';assert sha256(ckpt)==source['runtime']['checkpoints'][ckpt.name]
        model.load_state_dict(torch.load(ckpt,weights_only=False)['model']);model.eval();models[seed]=model
        cached={}
        for name in ('train','inner'):
            d=exp.grid_features(a,name,False)
            with torch.no_grad():
                residual=torch.cat([model(d['x'][i:i+4])[0] for i in range(0,len(d['x']),4)])
                mu=(d['base']+residual).cpu().numpy()
            cached[name]=dict(mu=mu[:,6:]*height,features=correction_features(a,name,mu),
                target=d['y'][:,6:].cpu().numpy(),mask=d['mask'][:,6:].cpu().numpy())
            del d
        t=cached['train'];iv=cached['inner'];mask=t['mask'][:,:,fit]
        features=t['features'][:,:,fit]*height[fit][None,None,:,None]
        scales=np.maximum(features[mask].std(0),.1)
        station_count=mask.sum((0,1));weight=np.broadcast_to(1/np.maximum(station_count,1),mask.shape)[mask];weight/=weight.sum()
        target=(t['target'][:,:,fit]-t['mu'][:,:,fit])[mask]
        selections[seed]={}
        for arm,columns in [('bias',0),('current',6),('history',18)]:
            X=design(t['features'],height,fit,columns,scales)[mask].astype('f8')
            Xinner=design(iv['features'],height,inner,columns,scales)
            mat=X.T@(X*weight[:,None]);rhs=X.T@(target*weight)
            penalty=np.eye(X.shape[-1]);penalty[0,0]=0
            best=float('inf');candidates=[]
            for ridge in ([0.] if columns==0 else [.0001,.001,.01,.1,1.]):
                beta=np.linalg.solve(mat+ridge*penalty,rhs)
                pred=iv['mu'][:,:,inner]+Xinner@beta
                score=point_score(pred,iv['target'][:,:,inner],iv['mask'][:,:,inner])
                candidates.append(dict(ridge=ridge,inner=score))
                if score['macro_rmse']<best:best=score['macro_rmse'];chosen=dict(columns=columns,scales=scales.tolist(),beta=beta.tolist(),ridge=ridge,inner=score)
            selections[seed][arm]=dict(**chosen,candidates=candidates)
        print(json.dumps(dict(stage='fit',seed=seed,inner={k:v['inner']['macro_rmse'] for k,v in selections[seed].items()})),flush=True)
    write_json(ROOT/'selection.lock.json',selections)
    results={}
    for seed in (42,17):
        model=models[seed];gen=GridRandom(True).cuda();ckpt=SOURCE/f'single_{seed}_1400.pt'
        assert sha256(ckpt)==source['runtime']['checkpoints'][ckpt.name]
        gen.load_state_dict(torch.load(ckpt,weights_only=False)['model']);results[str(seed)]={}
        for name in ('confirmation','reserved'):
            d=exp.grid_features(a,name,False)
            with torch.no_grad():res,h=model(d['x']);mu=d['base']+res;d['mu']=mu;d['h']=h
            feat=correction_features(a,name,mu.detach().cpu().numpy())
            _,p=exp.evaluate(gen,d,outer);_,p128=exp.evaluate(gen,d,outer,noise_seed=11703,members=128)
            mean=(mu[:,6:,outer]*d['height'][outer]).detach().cpu().numpy()
            y=p['target'];m=p['mask'];results[str(seed)][name]={}
            for arm in ('none','bias','current','history'):
                if arm=='none':change=np.zeros_like(mean)
                else:
                    sel=selections[seed][arm]
                    change=design(feat,height,outer,sel['columns'],np.array(sel['scales']))@np.array(sel['beta'])
                pred=mean+change;s=p['samples']+change[:,None];s128=p128['samples']+change[:,None]
                results[str(seed)][name][arm]=dict(point=point_score(pred,y,m),k64=probability_score(s,y,m),k128=probability_score(s128,y,m))
                np.savez_compressed(ROOT/f'{arm}_{seed}_{name}.npz',prediction=pred,samples=s,target=y,mask=m,station_ids=outer,hours=a[name+'_hours'][:,6:])
            del d
    write_json(ROOT/'results.json',dict(results=results,runtime=dict(seconds=time.monotonic()-started,code_sha256=sha256(__file__),
        selection_sha256=sha256(ROOT/'selection.lock.json'))))
    print(json.dumps(dict(stage='complete',seconds=time.monotonic()-started,results=results)),flush=True)


if __name__=='__main__':main()
