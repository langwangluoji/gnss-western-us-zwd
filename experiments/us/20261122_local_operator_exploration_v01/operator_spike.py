"""Throwaway B-only operator screening; no remote jobs or C/D labels."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256
from gnss_field.data import PackedStore
from gnss_field.evaluation import point_metrics
from gnss_field.training.baselines import _idw
from gnss_field.training.prediction import safe_query_geometry
from local_context_features import VertexQuery
from grid_pair_data import local_runtime

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
SOURCE=PROJECT/'results/20261122_local_coverage_expansion_v01'


def status(stage,**kw):
    write_json(ROOT/'status.json',dict(stage=stage,seconds=time.monotonic()-START,**kw));print(stage,kw,flush=True)


def query(v,indices,weights):
    return np.einsum('tnkc,nk->tnc',v[:,indices],weights,optimize=True) if v.ndim==3 else (v[:,indices]*weights[None]).sum(2)


def input_cache(a,settings,store):
    path=ROOT/'A_inputs.npz'
    if path.exists():
        if sha256(path)!=read_json(ROOT/'A_inputs.complete.json')['sha256']:raise ValueError('Input cache changed')
        return dict(np.load(path))
    ids=store.ids('A');nodes=np.arange(np.prod(store.shape));aq,_=safe_query_geometry(store,ids)
    q=VertexQuery(store.shape,aq,np.zeros(len(ids)),nodes)
    qa=q.indices.numpy();qw=q.weights.numpy();corners=np.unique(qa)
    lookup=np.searchsorted(corners,qa)
    raw=store.context_values(a['hours']).astype('f8');ok=np.isfinite(raw).all(-1)&store.query_valid[ids][None]
    m0=np.load(SOURCE/'cache/M0.npy',mmap_mode='r');xy=np.c_[store.grid['x'].ravel(),store.grid['y'].ravel()]
    ba=[];residual=[]
    for i,row in enumerate(raw):
        valid=np.isfinite(row).all(-1)
        idw=_idw(store.xy[ids][valid],row[valid,0],xy[corners],store.stats['input_mean'][0])
        mb=(m0[i].ravel()[qa]*qw).sum(1)
        base=mb+settings['baseline_weight']*((idw[lookup]*qw).sum(1)-mb)
        ba.append(base);residual.append(np.where(ok[i],row[:,0]-base,0.))
    result=dict(raw=raw,valid=ok,residual=np.array(residual),base_A=np.array(ba),A_indices=qa,A_weights=qw,
                A_xy=store.xy[ids],A_height=store.stations.height.to_numpy()[ids])
    np.savez_compressed(path,**result);write_json(ROOT/'A_inputs.complete.json',dict(sha256=sha256(path)))
    return result


def moment_features(inputs,xy,height):
    delta=inputs['A_xy'][None]-xy[:,None];dist=np.linalg.norm(delta,axis=-1)
    dh=inputs['A_height'][None]-height[:,None]
    residual=inputs['residual'].astype('f4');valid=inputs['valid'].astype('f4');columns=[]
    for radius in (50000.,100000.,200000.):
        for vertical in (None,750.):
            k=np.exp(-.5*(dist/radius)**2)*(dist<=3*radius)
            if vertical:k*=np.exp(-.5*(dh/vertical)**2)
            k=k.astype('f4');den=valid@k.T
            for moment in (np.ones_like(dist),delta[:,:,0]/radius,delta[:,:,1]/radius,dh/1000):
                value=(residual@(k*moment.astype('f4')).T)/np.maximum(den,1e-30)
                columns.append(value.astype('f4'))
    return np.stack(columns,-1)


def interactions(f):
    amp=f[:,:,59];innovation=f[:,:,59]-f[:,:,58]
    return np.stack([amp*f[:,:,56],amp*f[:,:,57],amp*f[:,:,53],amp*f[:,:,65],
                     innovation*f[:,:,56],innovation*f[:,:,57],amp*f[:,:,54],amp*f[:,:,55]],-1)


def metric(a,p,ix,role):return point_metrics(a['truth'][ix],p,a['valid'][ix]&role[None])


def ridge(a,features,split,scale):
    ti=np.searchsorted(a['hours'],split['train_hours']);ii=np.searchsorted(a['hours'],split['inner_hours'])
    center=features[ti].mean((0,1),dtype='f8');std=features[ti].std((0,1),dtype='f8');std[std<1e-4]=1.
    design=np.empty((len(a['hours']),len(a['station_ids']),features.shape[-1]+1),'f8')
    for start in range(0,len(design),16):
        f=(features[start:start+16].astype('f8')-center)/std
        f=np.concatenate([f,np.ones_like(f[...,:1])],-1)
        design[start:start+16]=query(f,a['query_indices'],a['query_weights'])
    fit=np.asarray(split['fit'],bool);inner=np.asarray(split['inner'],bool)
    mask=a['valid'][ti]&fit[None];counts=mask.sum(0);use=counts>0
    weights=np.where(mask,1/np.maximum(counts,1)[None]/use.sum(),0.)
    target=np.where(mask,(a['truth'][ti]-a['mean'][ti])/scale,0.)
    xx=design[ti][mask];ww=weights[mask];yy=target[mask]
    gram=xx.T@(ww[:,None]*xx);rhs=xx.T@(ww*yy);rows=[]
    for strength in (1e-4,.01,1.):
        beta=np.linalg.solve(gram+strength*np.eye(len(rhs)),rhs)
        assert np.max(abs((gram+strength*np.eye(len(rhs)))@beta-rhs))<1e-9
        p=a['mean'][ii]+scale*(design[ii]@beta)
        score=metric(a,p,ii,inner)
        rows.append(dict(ridge=strength,inner=score,beta=beta))
    best=min(rows,key=lambda r:r['inner']['station_macro_rmse_mm'])
    return dict(center=center,std=std,beta=best['beta'],ridge=best['ridge'],inner=best['inner'],
                prediction=(a['mean']+scale*(design@best['beta'])).astype('f4'),
                trials=[{k:v for k,v in r.items() if k!='beta'} for r in rows])


def lattice_kernel(inputs,store,bnodes,noise,vertical):
    corners=np.unique(inputs['A_indices']);ai=np.searchsorted(corners,inputs['A_indices'])
    qa=np.zeros((len(ai),len(corners)))
    for i in range(4):np.add.at(qa,(np.arange(len(ai)),ai[:,i]),inputs['A_weights'][:,i])
    xy=np.c_[store.grid['x'].ravel(),store.grid['y'].ravel()];h=store.grid['height'].ravel()
    def kernel(i,j):
        value=np.exp(-.5*(cdist(xy[i],xy[j])/100000.)**2)
        if vertical:value*=np.exp(-.5*((h[i,None]-h[None,j])/vertical)**2)
        return value
    saa=qa@kernel(corners,corners)@qa.T;cross=kernel(bnodes,corners)@qa.T
    out=np.zeros((len(inputs['valid']),len(bnodes)),'f4')
    patterns,inv=np.unique(inputs['valid'],axis=0,return_inverse=True)
    for i,valid in enumerate(patterns):
        rows=np.flatnonzero(inv==i);ids=np.flatnonzero(valid)
        if not len(ids):continue
        cov=saa[np.ix_(ids,ids)]+noise*np.eye(len(ids))
        alpha=cho_solve(cho_factor(cov,lower=True),inputs['residual'][np.ix_(rows,ids)].T)
        out[rows]=(cross[:,ids]@alpha).T
        if not np.isfinite(out[rows]).all():raise ValueError('Nonfinite assimilation')
    return out


def main():
    runtime=local_runtime();store=PackedStore(runtime['data_root'],purpose='context')
    immutable_json(ROOT/'experiment.lock.json',dict(code=sha256(__file__),plan=sha256(ROOT/'PLAN.md'),
        source_data=sha256(SOURCE/'data.npz'),source_split=sha256(SOURCE/'experiment.lock.json'),
        packed=store.manifest['fingerprint'],seed=42,kind='local exploratory spike',server_jobs=False,C=False,D=False))
    a=dict(np.load(SOURCE/'data.npz'));lock=read_json(SOURCE/'experiment.lock.json')
    settings=read_json(PROJECT/'results/20261120_local_context_three_arm_v01/dataset/settings.json');scale=settings['grid_scale']
    status('INPUTS');inputs=input_cache(a,settings,store)
    nodes=a['global_nodes'];xy=np.c_[store.grid['x'].ravel(),store.grid['y'].ravel()][nodes]
    moment_path=ROOT/'moments.npy'
    if not moment_path.exists():np.save(moment_path,moment_features(inputs,xy,store.grid['height'].ravel()[nodes]))
    moments=np.load(moment_path);x=np.concatenate([a['features'],a['innovation']/scale],-1)
    results={};predictions={};coefficients={}
    for fold,split in enumerate(lock['splits']):
        status('FITTING',fold=fold)
        rows={};predictions[fold]={};coefficients[fold]={}
        for name,features in [('linear',x),('interactions',np.concatenate([x,interactions(a['features'])],-1)),
                              ('directional',np.concatenate([x,moments/scale],-1))]:
            value=ridge(a,features,split,scale);predictions[fold][name]=value.pop('prediction')
            coefficients[fold][name]={k:v for k,v in value.items() if k in ('center','std','beta','ridge')}
            rows[name]={k:v for k,v in value.items() if k not in ('center','std','beta')}
            status('RIDGE_DONE',fold=fold,method=name,inner=rows[name]['inner']['station_macro_rmse_mm'])
        ti=np.searchsorted(a['hours'],split['train_hours']);ii=np.searchsorted(a['hours'],split['inner_hours'])
        mask=a['valid'][ti]&np.asarray(split['fit'],bool)[None];counts=mask.sum(0);use=counts>0
        weight=np.where(mask,1/np.maximum(counts,1)[None]/use.sum(),0.)
        variants=[]
        for vertical in (None,750.):
            for noise in (.1,1.):
                tag=f'kernel_h{vertical}_n{noise}';cache=ROOT/f'{tag}.npy'
                if not cache.exists():np.save(cache,lattice_kernel(inputs,store,nodes,noise,vertical))
                field=np.load(cache);delta=query(field,a['query_indices'],a['query_weights'])
                numerator=np.sum(weight*delta[ti]*np.where(mask,a['truth'][ti]-a['mean'][ti],0.))
                denominator=np.sum(weight*delta[ti]**2)
                amplitude=float(np.clip(numerator/max(denominator,1e-30),0,1.5))
                p=a['mean']+amplitude*delta
                variants.append(dict(vertical=vertical,noise=noise,amplitude=amplitude,inner=metric(a,p[ii],ii,np.asarray(split['inner'],bool)),prediction=p,tag=tag))
        best=min(variants,key=lambda r:r['inner']['station_macro_rmse_mm']);predictions[fold]['kernel']=best['prediction']
        coefficients[fold]['kernel']={k:best[k] for k in ('vertical','noise','amplitude','tag')}
        rows['kernel']=dict(inner=best['inner'],trials=[{k:v for k,v in r.items() if k!='prediction'} for r in variants])
        rows['zero']=dict(inner=metric(a,a['mean'][ii],ii,np.asarray(split['inner'],bool)));predictions[fold]['zero']=a['mean']
        winner=min(rows,key=lambda n:rows[n]['inner']['station_macro_rmse_mm'])
        results[str(fold)]=dict(selected=winner,rows=rows)
    # Publish decisions before looking at the outer labels in this run.
    immutable_json(ROOT/'selection.lock.json',dict(folds=results,coefficients=coefficients,selection='inner only; outer not used'))
    status('OUTER_EVALUATION')
    oi=np.searchsorted(a['hours'],lock['outer_hours'])
    from local_coverage_report import pair_diagnostics
    increments={}
    for fold,split in enumerate(lock['splits']):
        outer=np.asarray(split['outer'],bool)
        for name,p in predictions[fold].items():results[str(fold)]['rows'][name]['outer']=metric(a,p[oi],oi,outer)
        np.savez_compressed(ROOT/f'fold_{fold}_predictions.npz',hours=a['hours'],**predictions[fold])
        increments[str(fold)]=pair_diagnostics(a['truth'][oi],{n:p[oi] for n,p in predictions[fold].items()},a['valid'][oi]&outer[None],a['xy'],a['height'])
    write_json(ROOT/'results.json',results);write_json(ROOT/'increments.json',increments)
    status('COMPLETE',folds={f:dict(selected=r['selected'],inner=r['rows'][r['selected']]['inner']['station_macro_rmse_mm'],outer=r['rows'][r['selected']]['outer']['station_macro_rmse_mm']) for f,r in results.items()})


if __name__=='__main__':
    START=time.monotonic();main()
