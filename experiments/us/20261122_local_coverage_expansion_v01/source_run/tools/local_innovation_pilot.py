"""Local two-epoch point heads and Gaussian allocation audit; no remote actions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import lsq_linear
from scipy.spatial.distance import cdist
from scipy.special import ndtr
from scipy.stats import norm

from local_innovation_components import (InnovationHead, aggregate_innovation, nested_split,
    query_nodes, station_macro_mse, epoch_batches, gaussian_records, gaussian_objective, fit_gaussian)
from local_context_features import VertexQuery
from grid_pair_data import local_runtime
from grid_generation_metrics import score_ensemble
from gnss_field.artifacts import read_json, write_json, immutable_json, sha256, fingerprint
from gnss_field.data import PackedStore
from gnss_field.evaluation import point_metrics
from gnss_field.models.diffusion import keyed_noise
from gnss_field.training.engine import atomic_torch, capture_rng, restore_rng, PauseRequested
from gnss_field.training.prediction import safe_query_geometry
from gnss_field.training.baselines import _idw
from gnss_field.training.observable import _same
from gnss_field.runtime import configure

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'results/20261122_local_innovation_gaussian_v02'
SOURCE = PROJECT/'results/20261120_local_context_three_arm_v01'
SOURCE_BUNDLE = PROJECT/'results/20261119_local_edm_subspace_v01/server_bundle'
ARMS = ('constant', 'gated', 'free')


def status(stage, **details):
    write_json(ROOT/'status.json',dict(stage=stage,updated=time.time(),**details))
    print(stage,details,flush=True)


def prepare():
    ROOT.mkdir(exist_ok=True,parents=True)
    dataset=SOURCE/'dataset'; complete=read_json(dataset/'complete.json')
    cache=SOURCE/'M0_cache'; cache_meta=read_json(cache/'complete.json')
    deps={name:sha256(dataset/name) for name in complete['files']}
    if deps != complete['files'] or sha256(cache/'mean.npy')!=cache_meta['sha256']:
        raise ValueError('Previously verified feature or M0 cache changed')
    if sha256(local_runtime()['m0'])!='605650e000ce4a0ab31d3d85996fd16316c978547fe9f2821c65c346cd1b86f0':
        raise ValueError('Frozen M0 changed')
    bundle_manifest=read_json(SOURCE_BUNDLE/'manifest.json')
    if sha256(SOURCE_BUNDLE/'data.npz')!=bundle_manifest['data_sha256']:
        raise ValueError('Frozen probability bundle changed')
    a=dict(np.load(dataset/'arrays.npz')); settings=read_json(dataset/'settings.json')
    s=pd.read_csv(dataset/'stations.csv')
    original=read_json(PROJECT/'results/20261122_generation_postrun_research_v01/downloaded/data.lock.json')
    pool=a['hours'][a['core']]
    if len(pool)!=384 or not np.isin(pool,original['train_hours']).all():
        raise ValueError('Local direction pool not within allowed training hours')
    if not np.all(pool.astype('datetime64[h]').astype('datetime64[Y]')==np.datetime64('2018')):
        raise ValueError('Only B2018 is permitted')
    outer=a['hours'][a['outer']]
    registered=read_json(PROJECT/'results/20261120_postrun_optimization_diagnosis_v01/expanded_B_week_audit.json')
    weeks=np.array([np.datetime64(t,'h').astype('i8') for t in registered['week_starts']])
    if not np.any((outer[:,None]>=weeks)&(outer[:,None]<weeks+168),axis=1).all():
        raise ValueError('Pilot outer hours left registered outer weeks')
    splits=[]
    for fold in (0,1):
        allowed=(a['folds']!=fold)&a['valid'][a['core']].any(0)
        split=nested_split(pool,allowed,s.physical_group.to_numpy(),s.dense_group.to_numpy(),42)
        ti=np.searchsorted(a['hours'],split['train_hours'])
        eligible=a['valid'][ti][:,split['fit']].sum(1)>=4
        split['train_hours']=split['train_hours'][eligible]
        split['outer']=(a['folds']==fold)&a['valid'][a['outer']].any(0)
        splits.append(split)
    contract=dict(sources=deps,M0_cache=cache_meta['sha256'],bundle=sha256(SOURCE_BUNDLE/'manifest.json'),
        code={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('local_innovation_components.py')]},
        role='B2018 incremental development; historical frozen base/prior exposure',
        small_pool=pool,outer_hours=outer,outer_not_previous96=True,splits=splits,
        epochs=2,batch_hours=8,seed=42,ema=.999,warmup_fraction=.05,
        C2019=False,D2020=False,server_submissions=False)
    immutable_json(ROOT/'experiment.lock.json',contract)
    if (ROOT/'data_complete.json').exists():
        lock=read_json(ROOT/'data_complete.json')
        if sha256(ROOT/'data.npz')!=lock['sha256']: raise ValueError('Local data changed')
        return
    # Pointwise heads only require the exact four interpolation corners, not
    # interpolated inputs; all full-field publication operators remain intact.
    local_nodes=np.unique(a['query_indices'])
    global_nodes=a['nodes'][local_nodes]
    query_indices=np.searchsorted(local_nodes,a['query_indices'])
    raw=np.load(dataset/'features67.npy',mmap_mode='r')
    features=np.array(raw[:,local_nodes],dtype='f4')
    density=[5,11,17,23,29,35,41,47,49,51]
    features[:,:,density]=np.log1p(features[:,:,density])/np.log1p(174.)
    m0=np.load(cache/'mean.npy',mmap_mode='r')
    cache_hours=np.asarray(read_json(cache/'contract.json')['hours'])
    if not np.array_equal(cache_hours,a['hours']): raise ValueError('M0 time join changed')
    store=PackedStore(local_runtime()['data_root'],purpose='context')
    aid=store.ids('A'); aq,_=safe_query_geometry(store,aid)
    qA=VertexQuery(store.shape,aq,np.zeros(len(aid)),np.arange(np.prod(store.shape)))
    a_corners=np.unique(qA.indices.numpy()); grid=store.grid
    grid_xy=np.c_[grid['x'].ravel(),grid['y'].ravel()]
    distances=cdist(grid_xy[global_nodes],store.xy[aid])
    kernels=[np.where(distances<=3*s,np.exp(-.5*(distances/s)**2),0.) for s in (75000.,150000.)]
    J=np.zeros((len(a['hours']),len(global_nodes),2),'f4')
    base_nodes=np.empty((len(a['hours']),len(global_nodes)),'f4')
    maximum=0.; started=time.monotonic()
    for i,hour in enumerate(a['hours']):
        current=store.context_values([int(hour)])[0]
        valid=np.isfinite(current).all(1)
        idwA=_idw(store.xy[aid][valid],current[valid,0],grid_xy[a_corners],store.stats['input_mean'][0])
        aa=np.searchsorted(a_corners,qA.indices.numpy())
        idw_at_A=(idwA[aa]*qA.weights.numpy()).sum(-1)
        m0_at_A=qA(torch.tensor(np.asarray(m0[i]).reshape(-1))).numpy()
        base_A=m0_at_A+settings['baseline_weight']*(idw_at_A-m0_at_A)
        for k,kernel in enumerate(kernels):
            J[i,:,k]=aggregate_innovation(current[:,0],base_A,kernel,valid&store.query_valid[aid])
        base_nodes[i]=m0[i].ravel()[global_nodes]+settings['baseline_weight']*(features[i,:,59]*settings['grid_scale']-m0[i].ravel()[global_nodes])
        recovered=(base_nodes[i][query_indices]*a['query_weights']).sum(-1)
        maximum=max(maximum,float(np.max(abs(recovered-a['mean'][i]))))
        if (i+1)%96==0: status('PREPARING',hours=i+1,total=len(a['hours']),seconds=time.monotonic()-started)
    if maximum>.001: raise ValueError(f'Baseline grid query differs: {maximum}')
    bundle=dict(np.load(SOURCE_BUNDLE/'data.npz'))
    np.testing.assert_array_equal(bundle['station_ids'],a['station_ids'])
    arrays=dict(hours=a['hours'],features=features,innovation=J,query_indices=query_indices,
        query_weights=a['query_weights'],base_nodes=base_nodes,global_nodes=global_nodes,
        mean=a['mean'],truth=a['truth'],valid=a['valid'],amplitude=a['amplitude'],
        projection=bundle['projection'],remainder=bundle['remainder'],station_ids=a['station_ids'],
        xy=s[['x','y']].to_numpy(),height=s.height.to_numpy())
    np.savez_compressed(ROOT/'data.npz',**arrays)
    write_json(ROOT/'data_complete.json',dict(sha256=sha256(ROOT/'data.npz'),base_query_max_mm=maximum,
        hours=len(a['hours']),nodes=len(global_nodes),seconds=time.monotonic()-started))


class Pilot:
    def __init__(self):
        self.a=dict(np.load(ROOT/'data.npz')); self.lock=read_json(ROOT/'experiment.lock.json')
        self.settings=read_json(SOURCE/'dataset/settings.json')
        self.splits=[]
        for sp in self.lock['splits']:
            self.splits.append({k:np.asarray(v,dtype=bool if k in ('fit','inner','outer') else 'i8') for k,v in sp.items()})
        self.t={k:torch.tensor(self.a[k],device='cuda') for k in
                ('features','innovation','query_indices','query_weights','mean','truth','valid')}

    def indices(self,hours):
        ids=np.searchsorted(self.a['hours'],hours)
        if not np.array_equal(self.a['hours'][ids],hours): raise ValueError('Missing cached hour')
        return ids

    def predict_tensor(self,model,ix):
        correction=model(self.t['features'][ix],self.t['innovation'][ix])
        return self.t['mean'][ix]+query_nodes(correction,self.t['query_indices'],self.t['query_weights'])

    @torch.no_grad()
    def predict(self,model,hours):
        return np.concatenate([self.predict_tensor(model,ix).cpu().numpy() for ix in np.array_split(self.indices(hours),max(1,math.ceil(len(hours)/16)))])

    def metric(self,pred,hours,station_mask):
        ix=self.indices(hours)
        return point_metrics(self.a['truth'][ix],pred,self.a['valid'][ix]&station_mask[None])


def run_neural(data,arm,fold,directory,batches=None,stop_after=None,engineering=False):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    split=data.splits[fold]; hours=split['train_hours']
    batches=epoch_batches(hours,8,42,2) if batches is None else batches
    budget=len(batches); state_path=directory/'last.pt'
    signature=fingerprint(dict(data=sha256(ROOT/'experiment.lock.json'),arm=arm,fold=fold,batches=batches))
    torch.manual_seed(42); model=InnovationHead(arm,data.settings['grid_scale']).cuda()
    ema=copy.deepcopy(model); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    step=0; best_value=float('inf'); best_step=0; start=time.monotonic()
    if state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        if state['signature']!=signature: raise ValueError('Resume signature changed')
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema']); optimizer.load_state_dict(state['optimizer'])
        step=state['step']; best_value=state['best_value']; best_step=state['best_step']; restore_rng(state['rng'])
    ti=data.indices(hours); fit=torch.tensor(split['fit'],device='cuda')
    mask_all=data.t['valid'][ti]&fit[None]; counts=mask_all.sum(0)
    interval=max(1,math.ceil(math.ceil(len(hours)/8)/4))
    events={0,budget,200}|set(range(interval,budget+1,interval))
    if engineering: events={0,budget}
    def evaluate():
        nonlocal best_value,best_step
        hi=split['inner_hours']
        e=data.metric(data.predict(ema,hi),hi,split['inner'])
        r=data.metric(data.predict(model,hi),hi,split['inner'])
        value=e['station_macro_rmse_mm']
        row=dict(step=step,EMA=e,raw=r)
        with (directory/'development.jsonl').open('a',encoding='utf-8') as out: out.write(json.dumps(row)+'\n')
        if value<best_value:
            best_value=value; best_step=step
            atomic_torch(directory/'point_best.pt',dict(model=copy.deepcopy(ema.state_dict()),step=step,inner=e,signature=signature))
        return row
    if step==0: evaluate()
    stop=budget if stop_after is None else min(stop_after,budget)
    last_loss=None
    for j in range(step,stop):
        if (ROOT/'PAUSE').exists(): break
        ix=data.indices(batches[j]); warm=max(1,round(.05*budget))
        ratio=(j+1)/warm if j<warm else .01+.99*.5*(1+math.cos(math.pi*(j-warm)/max(1,budget-warm-1)))
        lr=1e-4*ratio
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        prediction=data.predict_tensor(model,ix)
        loss=station_macro_mse(prediction,data.t['truth'][ix],data.t['valid'][ix]&fit[None],counts,len(hours))
        if not torch.isfinite(loss): raise ValueError('Nonfinite MSE')
        loss.backward(); norm_grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm_grad): raise ValueError('Nonfinite gradient')
        optimizer.step()
        with torch.no_grad():
            for target,current in zip(ema.parameters(),model.parameters()): target.lerp_(current,.001)
        step=j+1; last_loss=float(loss.detach())
        with (directory/'training.jsonl').open('a',encoding='utf-8') as out:
            out.write(json.dumps(dict(step=step,loss=last_loss,learning_rate=lr,hours=len(ix),station_hours=int((data.t['valid'][ix]&fit[None]).sum()),seconds=time.monotonic()-start))+'\n')
        if step in events: evaluate()
        if step in events or step==stop:
            atomic_torch(state_path,dict(signature=signature,model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                step=step,best_value=best_value,best_step=best_step,rng=capture_rng()))
    # Save even when PAUSE arrived before the first update or between evaluation
    # boundaries; downstream callers must not treat this as a completed stage.
    atomic_torch(state_path,dict(signature=signature,model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
        step=step,best_value=best_value,best_step=best_step,rng=capture_rng()))
    write_json(directory/'status.json',dict(step=step,budget=budget,best_step=best_step,best_inner_RMSE=best_value,
        status='COMPLETE' if step==budget else 'PAUSED',seconds=time.monotonic()-start,last_loss=last_loss))
    return torch.load(state_path,map_location='cpu',weights_only=False)


def engineering(data):
    path=ROOT/'engineering.json'
    if path.exists(): return
    batches=[data.splits[0]['train_hours'][:8]]*200
    for arm in ARMS:
        continuous=run_neural(data,arm,0,ROOT/f'engineering/{arm}/continuous',batches,engineering=True)
        if continuous['step']!=200: raise PauseRequested('Continuous engineering run incomplete')
        half=run_neural(data,arm,0,ROOT/f'engineering/{arm}/split',batches,stop_after=100,engineering=True)
        if half['step']!=100: raise PauseRequested('Split engineering run incomplete')
        restored=run_neural(data,arm,0,ROOT/f'engineering/{arm}/split',batches,engineering=True)
        if restored['step']!=200: raise PauseRequested('Restored engineering run incomplete')
        for key in ('model','ema','optimizer','step','rng'): _same(continuous[key],restored[key])
    write_json(path,dict(passed=True,actual_training_path=True,continuous=200,resumed=[100,100],bitwise=True,arms=list(ARMS)))


def constant_exact(data,fold):
    split=data.splits[fold]; ix=data.indices(split['train_hours'])
    J=data.a['innovation'][ix]
    qj=J[:,data.a['query_indices']]*data.a['query_weights'][None,:,:,None]
    design=qj.sum(2)/2
    mask=data.a['valid'][ix]&split['fit'][None]; counts=mask.sum(0); use=counts>0
    w=np.broadcast_to(1/np.maximum(counts,1)[None]/use.sum(),mask.shape)[mask]
    x=design[mask]*np.sqrt(w[:,None]); y=(data.a['truth'][ix]-data.a['mean'][ix])[mask]*np.sqrt(w)
    result=lsq_linear(x,y,bounds=(-1,1),tol=1e-12)
    if not result.success: raise ValueError('Bounded constant fit failed')
    return result.x


def point_experiments(data):
    if (ROOT/'point_results.json').exists(): return read_json(ROOT/'point_results.json')
    result={}; outer_hours=np.asarray(data.lock['outer_hours'])
    for fold in (0,1):
        split=data.splits[fold]; directory=ROOT/f'fold_{fold}'; directory.mkdir(exist_ok=True)
        hi=split['inner_hours']; oi=data.indices(outer_hours)
        rows={}; predictions={}; models={}
        base_inner=data.a['mean'][data.indices(hi)]; base_outer=data.a['mean'][oi]
        rows['zero']=dict(inner=data.metric(base_inner,hi,split['inner']),outer=data.metric(base_outer,outer_hours,split['outer']),step=0)
        predictions['zero']=base_outer
        for arm in ARMS:
            status('TRAINING',fold=fold,arm=arm,hours=len(split['train_hours']))
            trained=run_neural(data,arm,fold,directory/arm)
            if trained['step']!=len(epoch_batches(split['train_hours'],8,42,2)):
                raise PauseRequested('Scientific point run incomplete; downstream stages blocked')
            state=torch.load(directory/arm/'point_best.pt',map_location='cuda',weights_only=False)
            model=InnovationHead(arm,data.settings['grid_scale']).cuda(); model.load_state_dict(state['model'])
            pred=data.predict(model,outer_hours); predictions[arm]=pred; models[arm]=model
            rows[arm]=dict(inner=state['inner'],outer=data.metric(pred,outer_hours,split['outer']),step=state['step'])
            np.savez_compressed(directory/f'{arm}_outer.npz',hours=outer_hours,prediction=pred,truth=data.a['truth'][oi],
                population=data.a['valid'][oi]&split['outer'][None],station_ids=data.a['station_ids'])
        beta=constant_exact(data,fold)
        # Direct bounded coefficients; no tanh optimization approximation here.
        J=data.a['innovation']; qj=(J[:,data.a['query_indices']]*data.a['query_weights'][None,:,:,None]).sum(2)/2
        exact=data.a['mean']+qj@beta
        rows['constant_exact']=dict(beta=beta.tolist(),inner=data.metric(exact[data.indices(hi)],hi,split['inner']),
            outer=data.metric(exact[oi],outer_hours,split['outer']),step=None)
        predictions['constant_exact']=exact[oi]
        # Selection only reads new spatial x weather inner, never outer scores.
        order=['zero','constant','gated','free']
        selected=min(order,key=lambda key:(rows[key]['inner']['station_macro_rmse_mm'],order.index(key)))
        if selected=='zero': all_mean=data.a['mean'].copy()
        elif selected=='constant_exact': all_mean=exact
        else: all_mean=data.predict(models[selected],data.a['hours'])
        np.savez_compressed(directory/'selected_mean.npz',hours=data.a['hours'],prediction=all_mean)
        write_json(directory/'point_selection.lock.json',dict(selected=selected,rows=rows,selection='new spatial x weather inner only',
            data=sha256(ROOT/'experiment.lock.json'),mean_sha256=sha256(directory/'selected_mean.npz')))
        result[str(fold)]=dict(selected=selected,rows=rows,train_hours=len(split['train_hours']),inner_hours=len(hi),
            train_stations=int(split['fit'].sum()),inner_stations=int(split['inner'].sum()),outer_hours=len(outer_hours))
        publication_check(data,fold,selected,models.get(selected),beta)
    write_json(ROOT/'point_results.json',result)
    return result


@torch.no_grad()
def publication_check(data,fold,selected,model,beta):
    source=dict(np.load(SOURCE/'dataset/arrays.npz')); raw=np.load(SOURCE/'dataset/features67.npy',mmap_mode='r')
    # Choose the first predeclared outer hour; regenerate J on all grid nodes
    # with the same input-only builder, rather than interpolating J or features.
    hour=int(data.lock['outer_hours'][0]); ix=int(data.indices([hour])[0])
    m0=np.load(SOURCE/'M0_cache/mean.npy',mmap_mode='r')[ix]
    from local_context_features import LocalFeatureBuilder
    from grid_pair_data import feature69
    from gnss_field.data.context import cycles
    store=PackedStore(local_runtime()['data_root'],purpose='context'); grid=store.grid; aid=store.ids('A')
    aq,_=safe_query_geometry(store,aid); nodes=np.arange(np.prod(store.shape)); xy=np.c_[grid['x'].ravel(),grid['y'].ravel()]
    builder=LocalFeatureBuilder(xy,store.xy[aid],store.stations.height.to_numpy()[aid],grid['height'],grid['region'],nodes,store.stats,aq,store.query_valid[aid])
    frames=np.array([store.context_values([hour-o])[0] for o in (0,6,12,18)])
    features=feature69(builder,frames,m0,cycles([hour])[0],store.shape)[:67].reshape(67,-1).T
    base=m0+data.settings['baseline_weight']*(features[:,59].reshape(store.shape)*data.settings['grid_scale']-m0)
    qA=VertexQuery(store.shape,aq,np.zeros(len(aid)),nodes)
    ba=qA(torch.tensor(base.ravel())).numpy(); valid=np.isfinite(frames[0]).all(1)&store.query_valid[aid]
    J=np.stack([aggregate_innovation(frames[0,:,0],ba,k,valid) for k in builder.kernels],-1).astype('f4')
    if selected=='zero': correction=np.zeros(len(nodes),'f4')
    elif selected=='constant_exact': correction=(J@beta/2).astype('f4')
    else: correction=model(torch.tensor(features[None],device='cuda'),torch.tensor(J[None],device='cuda'))[0].cpu().numpy()
    field=base+correction.reshape(store.shape)
    target=np.load(ROOT/f'fold_{fold}/selected_mean.npz')['prediction'][ix]
    bq,bh=safe_query_geometry(store,data.a['station_ids']); qB=VertexQuery(store.shape,bq,bh,nodes)
    actual=qB(torch.tensor(field.ravel())).numpy()
    error=float(np.max(abs(actual-target)))
    compact_feature_error=float(np.max(abs(features[data.a['global_nodes']]-data.a['features'][ix])))
    if error>.001 or compact_feature_error>1e-5: raise ValueError(f'Publication query mismatch: {error}, {compact_feature_error}')
    np.savez_compressed(ROOT/f'fold_{fold}/publication_case.npz',hour=hour,mean_Z0=field,base_Z0=base,
        correction_Z0=correction.reshape(store.shape),land=grid['region'],x=grid['x'],y=grid['y'])
    write_json(ROOT/f'fold_{fold}/publication_check.json',dict(max_query_error_mm=error,compact_feature_error=compact_feature_error))


def probability_metrics(data,mean,hours,station_mask,field_scale,nugget_scale,members=64):
    ix=data.indices(hours); a=data.a; tau=data.settings['tau']; ids=a['station_ids']
    z=keyed_noise(hours,members,20261124,shape=(a['projection'].shape[1],)).numpy()
    r=keyed_noise(hours,members,20262124,shape=(a['remainder'].shape[1],)).numpy()
    eta=keyed_noise(hours,members,20263124,shape=(905,)).numpy()[...,ids]
    field=z@a['projection'].T+r@a['remainder'].T
    samples=mean[:,None]+a['amplitude'][ix,None,None]*(field_scale*field+nugget_scale*tau*eta)
    # Slice to predeclared station role before forming geometry-based groups.
    use=np.flatnonzero(station_mask); pop=a['valid'][ix][:,use]
    scores=score_ensemble(a['truth'][ix][:,use],samples[:,:,use],mean[:,use],pop,a['xy'][use])
    variance=np.sum(a['projection']**2,axis=1)+np.sum(a['remainder']**2,axis=1)
    sd=a['amplitude'][ix,None]*np.sqrt(field_scale**2*variance[None]+nugget_scale**2*tau**2)
    error=np.where(a['valid'][ix]&station_mask[None],a['truth'][ix]-mean,0.)
    mask=a['valid'][ix]&station_mask[None]; count=mask.sum(0); keep=count>0
    macro=lambda value:float((np.where(mask,value,0.).sum(0)[keep]/count[keep]).mean())
    zz=error/sd
    crps=sd*(zz*(2*ndtr(zz)-1)+2*np.exp(-.5*zz*zz)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    scores.update(analytic_CRPS_mm=macro(crps),analytic_coverage90=macro(abs(error)<=norm.ppf(.95)*sd),analytic_width90_mm=macro(2*norm.ppf(.95)*sd))
    return scores


def gaussian_experiments(data):
    if (ROOT/'gaussian_results.json').exists(): return read_json(ROOT/'gaussian_results.json')
    output={}; outer_hours=np.asarray(data.lock['outer_hours'])
    cov=data.a['projection'].astype('f8')@data.a['projection'].astype('f8').T+data.a['remainder'].astype('f8')@data.a['remainder'].astype('f8').T
    _,groups=np.unique(np.floor(data.a['xy']/200000).astype(int),axis=0,return_inverse=True)
    for fold in (0,1):
        status('CALIBRATING',fold=fold)
        split=data.splits[fold]; mean=np.load(ROOT/f'fold_{fold}/selected_mean.npz')['prediction']
        train=data.indices(split['train_hours']); hi=split['inner_hours']; oi=data.indices(outer_hours)
        residual=data.a['truth'][train]-mean[train]
        mask=data.a['valid'][train]&split['fit'][None]
        records=gaussian_records(residual,mask,data.a['amplitude'][train],cov,groups)
        fits={kind:fit_gaussian(records,data.settings['tau'],kind) for kind in ('field_only','common','two_scales')}
        fits['original']=dict(field=1.,nugget=1.,NLL=gaussian_objective(np.zeros(2),records,data.settings['tau'])[0])
        for kind,row in fits.items():
            row['inner']=probability_metrics(data,mean[data.indices(hi)],hi,split['inner'],row['field'],row['nugget'])
        order=['original','field_only','common','two_scales']
        winner=min(order,key=lambda k:(fits[k]['inner']['group_ES_mm'],order.index(k)))
        # The winner is locked before outer values are computed.
        immutable_json(ROOT/f'fold_{fold}/probability_selection.lock.json',dict(selected=winner,fits=fits,selection='new inner group ES K64',seed=20261124))
        for kind,row in fits.items(): row['outer']=probability_metrics(data,mean[oi],outer_hours,split['outer'],row['field'],row['nugget'])
        grid=np.linspace(.1,1.8,41)
        profile=np.array([[gaussian_objective(np.log([f,n]),records,data.settings['tau'])[0] for n in grid] for f in grid])
        # Training-only synchronized weather-block and 50-km station-group bootstrap.
        rng=np.random.default_rng(42); boot=[]; train_hours=split['train_hours']
        blocks=(train_hours-train_hours.min())//168
        source_stations=pd.read_csv(SOURCE/'dataset/stations.csv')
        dense=source_stations.dense_group.to_numpy(); unique_dense=np.unique(dense[split['fit']])
        for _ in range(100):
            selected_blocks=rng.choice(np.unique(blocks),len(np.unique(blocks)),replace=True)
            rows=np.concatenate([np.flatnonzero(blocks==b) for b in selected_blocks])
            selected_dense=rng.choice(unique_dense,len(unique_dense),replace=True)
            columns=np.concatenate([np.flatnonzero((dense==g)&split['fit']) for g in selected_dense])
            # Duplicated station groups are separate composite groups, not two
            # noisy copies of the same random variable in a covariance matrix.
            br=[]
            for occurrence,g in enumerate(selected_dense):
                cols=np.flatnonzero((dense==g)&split['fit'])
                if len(cols)<2: continue
                rr=gaussian_records(residual[rows][:,cols],mask[rows][:,cols],data.a['amplitude'][train][rows],cov[np.ix_(cols,cols)],np.zeros(len(cols),int))
                br.extend(rr)
            if not br: continue
            total=sum(r['weight'] for r in br)
            for r in br:r['weight']/=total
            try:
                fitted=fit_gaussian(br,data.settings['tau'],'two_scales'); boot.append([fitted['field'],fitted['nugget']])
            except ValueError: continue
        np.savez_compressed(ROOT/f'fold_{fold}/gaussian_profile.npz',scales=grid,NLL=profile,bootstrap=np.asarray(boot))
        output[str(fold)]=dict(selected=winner,fits=fits,bootstrap_replicates=len(boot),
            bootstrap_scope='training-only coarse weather blocks x dense station groups; composite grouping differs from main 200km fit',
            scale_percentiles=np.quantile(boot,[.025,.5,.975],axis=0).tolist() if boot else None)
    write_json(ROOT/'gaussian_results.json',output)
    return output


def main():
    configure(type('R',(),{'cpus':2})(),42)
    prepare(); data=Pilot(); engineering(data)
    points=point_experiments(data); gaussian=gaussian_experiments(data)
    status('COMPLETE',device=torch.cuda.get_device_name(),point_selected={k:v['selected'] for k,v in points.items()},
        probability_selected={k:v['selected'] for k,v in gaussian.items()},server_jobs=False)


if __name__=='__main__':
    try: main()
    except PauseRequested as exc:
        status('PAUSED',reason=str(exc),server_jobs=False)
