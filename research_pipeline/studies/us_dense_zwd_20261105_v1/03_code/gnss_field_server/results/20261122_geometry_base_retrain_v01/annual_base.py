"""Full2019 frozen-base validation. No coefficient fitting or C-dependent input construction."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import time
import numpy as np
import torch
from retrain_common import ROOT,PROJECT,OPERATORS
from retrain_geometry import GeometryBase,lock_mean
from grid_pair_data import local_runtime,row_digest,verify_row
from local_context_features import VertexQuery
from gnss_field.data import PackedStore,ContextBuilder
from gnss_field.training.prediction import safe_query_geometry
from gnss_field.models.mean import MeanField
from gnss_field.training.fields import predict_patches
from gnss_field.evaluation import point_metrics
from gnss_field.artifacts import read_json,write_json,sha256,immutable_json


def query(values,q):return (values[...,q.indices.numpy()]*q.weights.numpy()).sum(-1)


def setup():
    runtime=local_runtime();store=PackedStore(runtime['data_root'],purpose='context');ids=store.ids('C');allnodes=np.arange(np.prod(store.shape))
    xy,height=safe_query_geometry(store,ids);xy=np.minimum(xy,np.array([store.shape[1]-1-1e-5,store.shape[0]-1-1e-5]))
    fullq=VertexQuery(store.shape,xy,height,allnodes);geo=GeometryBase(store,np.unique(fullq.indices.numpy()))
    return runtime,store,ids,geo,VertexQuery(store.shape,xy,height,geo.nodes)


def engineering(store,geo):
    errors=[]
    for name in ('C2019_confirmation','C2019_replication'):
        path=OPERATORS/name;hours=np.load(path/'station_predictions.npz')['hours'];inputs=np.load(path/'inputs.npz');published=np.load(path/'published_fields.npz')
        for i in (0,20):
            h=int(hours[i]);frames=np.array([store.context_values([h-offset])[0] for offset in (0,6,12,18)])
            new,_=geo.predict(frames,inputs['M0'][i].reshape(store.shape),h)
            reference=np.array([published[f'directional{f}'][i,geo.nodes] for f in (0,1)])
            error=float(np.max(abs(new-reference)));errors.append(error)
            if error>.001:raise ValueError(f'Query-corner readout differs from full field: {error}')
    write_json(ROOT/'mean_equivalence.json',dict(passed=True,max_full_vs_pruned_mm=max(errors),cases=4,mean_lock=sha256(ROOT/'mean.lock.json')))


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    lock_mean();runtime,store,ids,geo,q=setup();engineering(store,geo)
    previous=PROJECT/'results/20261119_frozen_m0_audit';manifest=read_json(previous/'station_prediction_manifest.json')
    np.testing.assert_array_equal(ids,manifest['year2019_station_ids'])
    model=MeanField().cuda().eval().requires_grad_(False);model.load_state_dict(torch.load(runtime['m0'],map_location='cuda',weights_only=False)['ema']);builder=ContextBuilder(store)
    target=ROOT/'annual2019';target.mkdir(exist_ok=True)
    immutable_json(target/'lock.json',dict(mean=sha256(ROOT/'mean.lock.json'),runner=sha256(__file__),stations=ids,
        old_prediction_manifest=sha256(previous/'station_prediction_manifest.json'),role='C2019 full exposed development validation; no fit',D2020=False))
    records=[];started=time.monotonic();max_m0=0.;n_done=0
    for month in range(1,13):
        name=f'2019-{month:02d}';directory=target/name;directory.mkdir(exist_ok=True)
        oldpath=previous/'monthly'/f'{name}.npz'
        with np.load(oldpath) as z:hours=z['hours'].copy()
        shape=(len(hours),)
        def mmap(filename,shape,dtype='f4'):
            path=directory/filename
            if path.exists():
                a=np.load(path,mmap_mode='r+')
                if a.shape!=shape:raise ValueError('Annual cache shape differs')
                return a
            return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
        grids=mmap('M0_Z0.npy',(*shape,*store.shape));means=mmap('new_means.npy',(*shape,2,len(ids)))
        base=mmap('old_base.npy',(*shape,len(ids)));m0c=mmap('M0_C.npy',(*shape,len(ids)));hashes=mmap('row_hashes.npy',(*shape,32),'u1')
        cursor=read_json(directory/'cursor.json') if (directory/'cursor.json').exists() else dict(done=0)
        for i in range(cursor['done']):verify_row([grids[i],means[i],base[i],m0c[i]],hashes[i])
        for start in range(cursor['done'],len(hours),4):
            if (ROOT/'PAUSE').exists():raise RuntimeError('Paused before annual batch; cache retained')
            h=hours[start:start+4]
            with torch.inference_mode():full=predict_patches(model,builder.build_context(h).to('cuda'),store,chunk=8)[0][:,0].cpu().numpy()
            frames=[store.context_values(h-offset) for offset in (0,6,12,18)]
            for j,hour in enumerate(h):
                i=start+j;field,b=geo.predict(np.array([v[j] for v in frames]),full[j],int(hour))
                grids[i]=full[j];means[i]=query(field,q);base[i]=query(b,q);m0c[i]=query(full[j].ravel()[geo.nodes],q)
                hashes[i]=row_digest([grids[i],means[i],base[i],m0c[i]])
            if start%96==0 or start+4>=len(hours):
                for a in (grids,means,base,m0c,hashes):a.flush()
                cursor['done']=min(start+4,len(hours));write_json(directory/'cursor.json',cursor)
                write_json(ROOT/'status.json',dict(stage='ANNUAL_BASE',month=month,hours_complete=n_done+cursor['done'],total_hours=8760,elapsed_seconds=time.monotonic()-started,training_started=False))
                print('ANNUAL',name,cursor['done'],len(hours),round(time.monotonic()-started,1),flush=True)
        # Predictions are completely committed before archived labels are accessed.
        commit=dict(means=sha256(directory/'new_means.npy'),base=sha256(directory/'old_base.npy'),M0=sha256(directory/'M0_C.npy'),grid=sha256(directory/'M0_Z0.npy'),rows=sha256(directory/'row_hashes.npy'))
        immutable_json(directory/'prediction_commit.json',commit)
        with np.load(oldpath) as z:
            truth=z['truth'].copy();pop=z['mask'].copy();coarse=z['coarse'].copy();idw=z['idw'].copy();ref=z['mean'].copy()
        error=float(np.max(abs(m0c[pop]-ref[pop])));max_m0=max(max_m0,error)
        if error>.001:raise ValueError(f'Annual M0 requery mismatch: {name}:{error}')
        arrays=dict(hours=hours,truth=truth,population=pop,coarse=coarse,idw=idw,M0=np.array(m0c),old_base=np.array(base),new0=np.array(means[:,0]),new1=np.array(means[:,1]))
        np.savez_compressed(directory/'scored.npz',**arrays);records.append(arrays);n_done+=len(hours)
    a={k:np.concatenate([r[k] for r in records]) for k in records[0]}
    scores={k:point_metrics(a['truth'],a[k],a['population']) for k in ('coarse','idw','M0','old_base','new0','new1')}
    if len(a['hours'])!=8760 or a['population'].sum()!=642258:raise ValueError('Annual scored population changed')
    gate=all(scores[f'new{f}']['station_macro_rmse_mm']<min(scores['old_base']['station_macro_rmse_mm'],scores['idw']['station_macro_rmse_mm']) for f in (0,1))
    result=dict(scores=scores,hours=8760,stations=76,station_hours=642258,M0_requery_max_mm=max_m0,gate_passed=gate,
        criterion='both frozen new fold means improve full-year macroRMSE over old mixture and A-IDW; no C retuning',D2020=False,seconds=time.monotonic()-started)
    write_json(ROOT/'annual_results.json',result);write_json(ROOT/'status.json',dict(stage='ANNUAL_PASSED' if gate else 'STOP_BASE_GATE',results=result))
    print('ANNUAL_RESULTS',result,flush=True)


if __name__=='__main__':main()
