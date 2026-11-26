"""Local research spike: frozen learned features and input-only innovations."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
import sys, json, time, hashlib
import numpy as np
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path[:0] = [str(PROJECT/'tools'), str(PROJECT/'src')]
from simple_weather_unet_pilot import PilotData, SimpleUNet, DATA, metrics
from gnss_field.models.conditioned_unet import ConditionedUNet
from gnss_field.data import PackedStore
from gnss_field.artifacts import write_json, read_json, immutable_json, sha256


def solve_ridge(features, target, mask, strength):
    """Station-macro MSE; inputs already queried in surface units."""
    x, y, mask = np.asarray(features,'f8'), np.asarray(target,'f8'), np.asarray(mask,bool)
    if not np.isfinite(x).all() or not np.isfinite(y[mask]).all():
        raise ValueError('Nonfinite scored data')
    counts = mask.sum(0)
    if not (counts > 0).any(): raise ValueError('Empty training population')
    w = mask / np.maximum(counts,1)[None] / (counts>0).sum()
    rms = np.sqrt(np.einsum('tn,tnc->c',w,x*x)).clip(1e-6)
    xx = x[mask]/rms
    ww, yy = w[mask], y[mask]
    gram = xx.T@(ww[:,None]*xx)
    rhs = xx.T@(ww*yy)
    matrix = gram+strength*np.eye(len(rhs))
    beta = np.linalg.solve(matrix,rhs)
    error = float(np.max(abs(matrix@beta-rhs)))
    if error>1e-8: raise ValueError('Ridge solve residual')
    return beta/rms,error


def kernels(axy, ah, xy, height, device='cuda'):
    delta = axy[None]-xy[:,None]
    distance = np.linalg.norm(delta,axis=-1)
    dh = ah[None]-height[:,None]
    out=[]
    for radius in (50000.,100000.,200000.):
        for vertical in (None,750.):
            k=np.exp(-.5*(distance/radius)**2)*(distance<=3*radius)
            if vertical: k*=np.exp(-.5*(dh/vertical)**2)
            moments=np.stack([np.ones_like(dh),delta[:,:,0]/radius,delta[:,:,1]/radius,dh/1000],0)
            out.append(torch.as_tensor(k[None]*moments,dtype=torch.float32,device=device))
    return out


def moments(residual,valid,kernel):
    columns=[]
    clean=torch.where(valid,residual,0.)
    for k in kernel:
        denominator=valid.float()@k[0].T
        numerator=torch.einsum('ba,cna->bnc',clean,k)
        columns.append(numerator/denominator.clamp_min(1e-20)[...,None])
    return torch.cat(columns,-1)


def features(model,x,wide):
    if wide:
        a,b,c=model.encoder(x)
        d=model.decoder.up2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        h=model.decoder.up1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return h,model.decoder.head(h)
    a=model.first(x); b=model.second(F.avg_pool2d(a,2)); c=model.bottom(F.avg_pool2d(b,2))
    d=model.up2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
    h=model.up1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
    return h,model.head(h)


def sources():
    return {f'{kind}_{seed}':PROJECT/'results'/folder/sub/'best.pt'
            for kind,folder,prefix in [('small','20261123_simple_weather_unet_v02','gnss_weather_terrain_seed'),
                                       ('wide','20261123_enhanced_encoder_generative_v01','encoder_seed')]
            for seed in (42,17) for sub in [prefix+str(seed)]}


@torch.no_grad()
def extract(model,wide,data,store,raw,out):
    n=len(data.hours); ns=len(data.a['heights'])
    qi=data.a['qindices']; nodes,inv=np.unique(qi,return_inverse=True); lookup=inv.reshape(qi.shape)
    xy=np.c_[store.grid['x'].ravel(),store.grid['y'].ravel()]
    aids=store.ids('A'); k=kernels(store.xy[aids],store.stations.height.to_numpy()[aids],xy[nodes],store.grid['height'].ravel()[nodes])
    valid=torch.tensor(np.isfinite(raw).all(-1)&store.query_valid[aids][None],device='cuda')
    obs=torch.tensor(np.nan_to_num(raw[:,:,0]),device='cuda')
    qw=data.qw; hf=data.height_factor
    prediction=np.empty((n,ns),'f4'); design=np.empty((n,ns,41),'f4')
    model.eval()
    maximum=0.
    for start in range(0,n,8):
        ix=np.arange(start,min(start+8,n)); x,base=data.batch(ix,True)
        h,delta=features(model,x,wide); field=(base+10*delta)[:,0]
        reference=(base+10*model(x))[:,0]
        maximum=max(maximum,float((reference-field).abs().max()))
        pred=data.query(field,np.arange(ns)); prediction[ix]=pred.cpu().numpy()
        # Innovations use public-height A values, never B labels.
        pa=(field.flatten(1)[:,data.qi[aids]]*data.qw[aids]).sum(-1)
        mf=moments(obs[ix]-pa,valid[ix],k)/10
        geo=(mf[:,lookup,:]*qw[None,:,:,None]).sum(2)*hf[None,:,None]
        deep=(h.flatten(2)[:,:,data.qi]*qw[None,None]).sum(-1).transpose(1,2)*hf[None,:,None]
        bias=hf[None,:,None].expand(len(ix),-1,-1)
        f=torch.cat([deep,bias,geo],-1)
        if not torch.isfinite(f).all() or not torch.isfinite(pred).all():raise ValueError('Nonfinite inference')
        design[ix]=f.cpu().numpy()
    np.savez(out/'features.npz',prediction=prediction,design=design)
    return prediction,design,maximum


def score(data,p,rows,ids):
    return metrics(p[np.ix_(rows,ids)],data.a['labels'][np.ix_(rows,ids)],data.a['mask'][np.ix_(rows,ids)])


def paired_interval(data,p,b,rows,ids,seed=20261123):
    months=data.hours[rows].astype('datetime64[h]').astype('datetime64[M]').astype(int)
    blocks=[np.flatnonzero(months==m) for m in np.unique(months)]
    y=data.a['labels'][np.ix_(rows,ids)]; mask=data.a['mask'][np.ix_(rows,ids)]
    x=p[np.ix_(rows,ids)]; z=b[np.ix_(rows,ids)]
    rng=np.random.default_rng(seed); values=[]
    for _ in range(2000):
        ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
        values.append(metrics(x[ix],y[ix],mask[ix])['macro_rmse']-metrics(z[ix],y[ix],mask[ix])['macro_rmse'])
    return np.quantile(values,[.025,.975]).tolist()


def main():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    data=PilotData();store=PackedStore(DATA,purpose='context')
    src=sources()
    immutable_json(ROOT/'experiment.lock.json',dict(code=sha256(__file__),plan=sha256(ROOT/'PLAN.md'),
        sources={k:sha256(p) for k,p in src.items()},data=sha256(PROJECT/'results/20261123_simple_weather_unet_v02/data.npz'),
        candidates=['original','deep','geo6','geo24','deep_geo24'],ridge=[.0001,.01,1.],
        selection='inner weather and inner B only',D_read=False,server_jobs=False))
    raw=store.context_values(data.hours).astype('f4')
    tr=data.rows('train');inner=data.rows('inner'); fit=data.a['split_fit']; vi=data.a['split_inner']
    populations={'fit':(tr,fit),'inner':(inner,vi),'outer':(data.rows('outer'),data.a['split_outer']),
                 'temporal':(data.rows('outer'),fit),'C48':(data.rows('development'),data.a['C_ids'])}
    cols={'deep':list(range(17)), 'geo6':list(range(17,41,4)), 'geo24':list(range(17,41)), 'deep_geo24':list(range(41))}
    results={}
    for name,path in src.items():
        out=ROOT/name;out.mkdir(exist_ok=True)
        if (out/'results.json').exists():results[name]=read_json(out/'results.json');continue
        wide=name.startswith('wide');model=(ConditionedUNet() if wide else SimpleUNet()).cuda()
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model'])
        if (out/'features.npz').exists():
            f=np.load(out/'features.npz');base,design=f['prediction'],f['design'];err=0.
        else:base,design,err=extract(model,wide,data,store,raw,out)
        trials={'original':dict(inner=score(data,base,inner,vi),columns=[],beta=[])}
        yy=(data.a['labels'][np.ix_(tr,fit)]-base[np.ix_(tr,fit)])/10
        mask=data.a['mask'][np.ix_(tr,fit)]
        for method,c in cols.items():
            all_trials=[]
            for strength in (.0001,.01,1.):
                beta,res=solve_ridge(design[np.ix_(tr,fit,c)],yy,mask,strength)
                pred=base+10*(design[:,:,c]@beta)
                m=score(data,pred,inner,vi)
                all_trials.append(dict(inner=m,columns=c,beta=beta,ridge=strength,normal_error=res))
            chosen=min(all_trials,key=lambda v:v['inner']['macro_rmse'])
            trials[method]={**chosen,'trials':[{k:v for k,v in t.items() if k!='beta'} for t in all_trials]}
        selected=min(trials,key=lambda v:trials[v]['inner']['macro_rmse'])
        immutable_json(out/'selection.lock.json',dict(selected=selected,candidates=trials,feature_path_error=err))
        result=dict(selected=selected,models={})
        predictions={}
        for method,t in trials.items():
            pred=base if method=='original' else base+10*(design[:,:,t['columns']]@np.asarray(t['beta']))
            predictions[method]=pred
            result['models'][method]={p:score(data,pred,*args) for p,args in populations.items()}
        chosen=predictions[selected]
        result['outer_delta']=result['models'][selected]['outer']['macro_rmse']-result['models']['original']['outer']['macro_rmse']
        result['outer_ci']=paired_interval(data,chosen,base,*populations['outer'])
        result['C48_delta']=result['models'][selected]['C48']['macro_rmse']-result['models']['original']['C48']['macro_rmse']
        np.savez(out/'predictions.npz',**predictions,hours=data.hours)
        write_json(out/'results.json',result);results[name]=result
        write_json(ROOT/'status.json',dict(stage='FITTING',completed=list(results),seconds=time.monotonic()-started))
        print(json.dumps(dict(source=name,selected=selected,outer_delta=result['outer_delta'],outer_ci=result['outer_ci'],C48_delta=result['C48_delta'])),flush=True)
        del model;torch.cuda.empty_cache()
    assert all(sha256(p)==read_json(ROOT/'experiment.lock.json')['sources'][k] for k,p in src.items())
    write_json(ROOT/'results.json',results)
    write_json(ROOT/'status.json',dict(stage='COMPLETE',seconds=time.monotonic()-started,source_weights_unchanged=True,server_jobs=False))


if __name__=='__main__':main()
