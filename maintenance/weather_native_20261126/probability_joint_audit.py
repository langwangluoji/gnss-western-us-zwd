"""CPU joint scores of saved IID station-trajectory ensembles; no model fitting."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'


def fair_energy(samples,truth):
    # x[K,origin,vector], y[origin,vector]. Normalize Euclidean norm by sqrt(D).
    # Missing observation coordinates are omitted identically in both terms.
    K=len(samples)
    if K<2:raise ValueError('at least two IID members are required')
    scores=[];i,j=np.triu_indices(K,1)
    for start in range(0,len(truth),64):
        y=np.asarray(truth[start:start+64],dtype='float64');valid=np.isfinite(y)
        x=np.where(valid[None],samples[:,start:start+64],0).astype('float64');y=np.nan_to_num(y)
        count=valid.sum(-1);distance=np.sqrt(((x-y[None])**2).sum(-1)).mean(0)
        norm=(x*x).sum(-1).T
        product=np.matmul(x.transpose(1,0,2),x.transpose(1,2,0))
        pair=np.sqrt(np.maximum(norm[:,:,None]+norm[:,None,:]-2*product,0))[:,i,j].sum(-1)/(K*(K-1))
        scores.append(np.divide(distance-pair,np.sqrt(count),out=np.full(count.shape,np.nan),where=count>0))
    return np.concatenate(scores)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);args=p.parse_args();sys.path.insert(0,str(ENGINE))
    from tools.terrain_diffusion.native_paths import file_path
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,file_path(value))
    from tools.terrain_diffusion.native_weather_evaluation import fair_crps
    from tools.terrain_diffusion.data import sha,write_json
    lineage=json.loads((args.evaluation/'lineage.json').read_text(encoding='utf-8'))
    if not lineage['independent_iid_member_noise']:raise ValueError('fair score requires IID members')
    if sha(args.evaluation/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('evaluation source changed')
    if sha(args.cache/'manifest.json')!=lineage['cache_manifest_sha256']:raise ValueError('different calendar')
    with np.load(args.evaluation/'evaluation.npz') as data:
        samples=data['samples'];truth=data['truth']
    K,L,M,T,Q,H=samples.shape
    geometry=json.loads((args.cache/'geometry.json').read_text(encoding='utf-8'));st=np.asarray(geometry['dev_static'])
    if Q!=len(st):raise ValueError('query geometry differs')
    scores=fair_energy(samples.reshape(K,L*M*T,Q*H),truth.reshape(L*M*T,Q*H)).reshape(L,M,T)
    energy=dict(mean_mm=float(np.nanmean(scores)),monthly_mean_mm=np.nanmean(scores,axis=(0,2)).tolist(),
        meaning='fair energy of the whole 0..6h station vector; Euclidean norm divided by sqrt(valid observed dimension)')
    i,j=np.triu_indices(Q,1);distance=np.linalg.norm((st[i,:2]-st[j,:2])*1000,axis=-1);pairs={}
    for lo,hi in [(0,50),(50,150),(150,300),(300,np.inf)]:
        a=i[(distance>=lo)&(distance<hi)];b=j[(distance>=lo)&(distance<hi)]
        record=dict(pair_count=len(a))
        for horizon in (0,6):
            sums=np.zeros(M);counts=np.zeros(M,dtype='int64')
            for start in range(0,len(a),32):
                aa=a[start:start+32];bb=b[start:start+32]
                value=samples[...,aa,horizon]-samples[...,bb,horizon]
                target=truth[...,aa,horizon]-truth[...,bb,horizon]
                score=fair_crps(value,target)
                sums+=np.nansum(score,axis=(0,2,3));counts+=np.isfinite(score).sum(axis=(0,2,3))
            record[f'h{horizon}']=dict(fair_crps_mm=float(sums.sum()/counts.sum()) if counts.sum() else None,
                monthly_score_sum=sums.tolist(),monthly_valid_count=counts.tolist())
        pairs[f'{lo}-{hi:g}']=record
    # Fixed forecast-origin trajectories support this test. Independently sampled
    # h0 fields from successive origins are not a continuous member trajectory.
    increment=fair_crps(np.diff(samples,axis=-1),np.diff(truth,axis=-1))
    metrics=json.loads((args.evaluation/'metrics.json').read_text(encoding='utf-8'))
    increment_value=float(np.nanmean(increment))
    np.testing.assert_allclose(increment_value,metrics['forecast_increment_fair_crps_mm'],atol=1e-10,rtol=0)
    payload=dict(energy=energy,spatial_pair_crps=pairs,within_forecast_increment_fair_crps_mm=increment_value,
        members=K,observation_role='same exposed 2020 development calendar; no fitting',
        current_fields_across_forecast_origins_are_not_a_single_generated_member_trajectory=True,
        source_lineage_sha256=sha(args.evaluation/'lineage.json'),code_sha256=sha(Path(__file__)),
        strict_oof=False,confirmation_C_D_read=False)
    args.out.mkdir(parents=True,exist_ok=True);write_json(args.out/'joint_metrics.json',payload)
    print(json.dumps(dict(energy_mm=energy['mean_mm'],within_forecast_increment_crps_mm=increment_value,
        spatial_pair_crps={key:{h:v[h]['fair_crps_mm'] for h in ('h0','h6')} for key,v in pairs.items()})))


if __name__=='__main__':main()
