"""Locked C2019 development check after B-only fits. D remains sealed."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import time
import numpy as np
import torch
from operator_spike import ROOT,SOURCE,PROJECT,moment_features,lattice_kernel,interactions,query
from gate_spike import Gate,scene_features
from gnss_field.artifacts import read_json,write_json,immutable_json,sha256
from gnss_field.data import PackedStore
from gnss_field.data.context import cycles
from gnss_field.evaluation import point_metrics
from gnss_field.geometry import coarsen_100km
from gnss_field.training.prediction import safe_query_geometry
from local_context_features import LocalFeatureBuilder,VertexQuery
from local_coverage_data import compact67
from local_innovation_components import aggregate_innovation
from grid_pair_data import local_runtime,GridStudy
from local_coverage_report import pair_diagnostics
from gnss_field.runtime import configure


def main():
    started=time.monotonic();configure(type('R',(),{'cpus':2})(),42)
    dest=ROOT/'C2019_confirmation';dest.mkdir(exist_ok=True)
    hours=np.array([np.datetime64(f'2019-{m:02d}-07T{h:02d}','h').astype('i8') for m in range(1,13) for h in (0,6,12,18)])
    selected=read_json(ROOT/'selection.lock.json');gate_lock=read_json(ROOT/'gate_selection.lock.json')
    # These files contain B-only fitted parameters. Their identities are frozen
    # before any archived C truth, support mask, or score is opened below.
    weights={f:ROOT/f'gate_fold_{f}/selected.pt' for f in (0,1)}
    immutable_json(dest/'protocol.lock.json',dict(code=sha256(__file__),hours=hours,
        source_selection=sha256(ROOT/'selection.lock.json'),gate_selection=sha256(ROOT/'gate_selection.lock.json'),
        gate_weights={str(f):sha256(p) for f,p in weights.items()},
        products=['base','linear0','linear1','directional0','directional1','kernel0','kernel1','gate0','gate1'],
        no_new_fitting=True,role='C2019 development confirmation; historically exposed development year, not blind',D2020=False,server_jobs=False,
        interpretation='B outer has informed prioritization; all frozen candidates reported, no C retuning'))
    runtime=local_runtime();store=PackedStore(runtime['data_root'],purpose='context');grid=store.grid
    ids=store.ids('A');allnodes=np.arange(np.prod(store.shape));xy=np.c_[grid['x'].ravel(),grid['y'].ravel()]
    aq,_=safe_query_geometry(store,ids);qa=VertexQuery(store.shape,aq,np.zeros(len(ids)),allnodes)
    aid=qa.indices.numpy();aw=qa.weights.numpy()
    maker=LocalFeatureBuilder(xy,store.xy[ids],store.stations.height.to_numpy()[ids],grid['height'],grid['region'],allnodes,store.stats,aq,store.query_valid[ids])
    settings=read_json(PROJECT/'results/20261120_local_context_three_arm_v01/dataset/settings.json');scale=settings['grid_scale']
    scratch=object.__new__(GridStudy);scratch.runtime=runtime;scratch.store=store;scratch.device='cuda';scratch.net=None
    cache=dest/'inputs.npz'
    if not cache.exists():
        fs=[];m0s=[];bases=[];js=[];residuals=[];raws=[];valids=[]
        for i,hour in enumerate(hours):
            m0=scratch.frozen_m0(int(hour))
            frames=np.array([store.context_values([int(hour)-o])[0] for o in (0,6,12,18)])
            f=compact67(maker,frames,m0,cycles([hour])[0])
            base=m0.ravel()+settings['baseline_weight']*(f[:,59]*scale-m0.ravel())
            raw=frames[0];valid=np.isfinite(raw).all(1)&store.query_valid[ids]
            ba=(base[aid]*aw).sum(1)
            J=np.stack([aggregate_innovation(raw[:,0],ba,k,valid) for k in maker.kernels],-1).astype('f4')
            fs.append(f);m0s.append(m0.ravel());bases.append(base);js.append(J)
            residuals.append(np.where(valid,raw[:,0]-ba,0.));raws.append(raw);valids.append(valid)
            if (i+1)%8==0:print('C_INPUTS',i+1,48,flush=True)
        np.savez_compressed(cache,features=np.array(fs),M0=np.array(m0s),base=np.array(bases),innovation=np.array(js),
            residual=np.array(residuals),raw=np.array(raws),valid=np.array(valids))
        write_json(dest/'inputs.complete.json',dict(sha256=sha256(cache)))
    assert sha256(cache)==read_json(dest/'inputs.complete.json')['sha256']
    a=dict(np.load(cache));inputs=dict(residual=a['residual'],valid=a['valid'],A_indices=aid,A_weights=aw,A_xy=store.xy[ids],A_height=store.stations.height.to_numpy()[ids])
    fields={'base':a['base'],'M0':a['M0']};moments=moment_features(inputs,xy,grid['height'].ravel())
    x=np.concatenate([a['features'],a['innovation']/scale],-1)
    for f in (0,1):
        coeff=selected['coefficients'][str(f)]
        for name,features in [('linear',x),('directional',np.concatenate([x,moments/scale],-1))]:
            row=coeff[name];beta=np.asarray(row['beta']);delta=[]
            for j in range(len(hours)):
                value=(features[j].astype('f8')-np.asarray(row['center']))/np.asarray(row['std'])
                delta.append(scale*(value@beta[:-1]+beta[-1]))
            fields[f'{name}{f}']=(a['base']+np.array(delta)).astype('f4')
        row=coeff['kernel'];field=lattice_kernel(inputs,store,allnodes,row['noise'],row['vertical'])
        fields[f'kernel{f}']=(a['base']+row['amplitude']*field).astype('f4')
        state=torch.load(weights[f],map_location='cpu',weights_only=False);model=Gate(np.zeros(8),np.ones(8),state['amplitude']['amplitude'])
        model.load_state_dict(state['model'])
        scene=scene_features(a['features']);delta=[]
        with torch.no_grad():
            for j in range(len(hours)):delta.append(model(torch.tensor(scene[j]),torch.tensor(field[j])).numpy())
        fields[f'gate{f}']=(a['base']+np.array(delta)).astype('f4')
    # Predictions for all fields are committed before opening any C labels.
    np.savez_compressed(dest/'published_fields.npz',hours=hours,**fields)
    write_json(dest/'prediction_commit.json',dict(fields_sha256=sha256(dest/'published_fields.npz'),before_C_labels=True))
    cids=store.ids('C');cq,height=safe_query_geometry(store,cids)
    edge=(cq[:,0]>=store.shape[1]-1)|(cq[:,1]>=store.shape[0]-1)
    cq=np.minimum(cq,np.array([store.shape[1]-1-1e-5,store.shape[0]-1-1e-5]))
    qc=VertexQuery(store.shape,cq,height,allnodes);ci=qc.indices.numpy();cw=qc.weights.numpy()
    prediction={name:query(v,ci,cw) for name,v in fields.items()}
    truth=[];pop=[];reference=[]
    legacy=PROJECT/'results/20261119_frozen_m0_audit/release/seed_42/mean_base/full2019/5000'
    for hour in hours:
        month=str(hour.astype('datetime64[h]').astype('datetime64[M]'))
        with np.load(legacy/f'forecast_{month}.npz') as z:
            np.testing.assert_array_equal(z['station_ids'],cids)
            ix=int(np.searchsorted(z['hours'],hour));assert z['hours'][ix]==hour
            truth.append(z['truth'][ix]);pop.append(z['population_F'][ix]&z['normal_support_S0'][ix]);reference.append(z['prediction'][ix])
    truth=np.array(truth);pop=np.array(pop);reference=np.array(reference)
    assert not (pop[:,edge]).any()
    error=float(np.max(abs(prediction['M0'][pop]-reference[pop])))
    if error>.001:raise ValueError(f'M0 differs from archived release: {error}')
    scores={name:point_metrics(truth,v,pop) for name,v in prediction.items()}
    coarse={name:query(coarsen_100km(torch.tensor(v.reshape(len(hours),*store.shape))).numpy().reshape(len(hours),-1),ci,cw) for name,v in fields.items()}
    coarse_scores={name:point_metrics(truth,v,pop) for name,v in coarse.items()}
    increments=pair_diagnostics(truth,prediction,pop,store.xy[cids],height)
    np.savez_compressed(dest/'station_predictions.npz',hours=hours,station_ids=cids,truth=truth,population=pop,**prediction)
    write_json(dest/'results.json',dict(scores=scores,coarsened_scores=coarse_scores,increments=increments,
        M0_reproduction_max_mm=error,station_hours=int(pop.sum()),stations=int(pop.any(0).sum()),hours=48,
        scope='fixed C2019 development dates; not full-year or final retrospective',D2020=False,seconds=time.monotonic()-started))
    print('C_CONFIRMATION_COMPLETE',dict(hours=48,stations=int(pop.any(0).sum()),M0_error=error,RMSE={k:v['station_macro_rmse_mm'] for k,v in scores.items()},seconds=time.monotonic()-started),flush=True)


if __name__=='__main__':main()
