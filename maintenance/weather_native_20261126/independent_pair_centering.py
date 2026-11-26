"""Disjoint-pair, IID-preserving mean centering of frozen complete field samples."""
import argparse
import json
from pathlib import Path
import numpy as np


def independent_pair_center(samples,base):
    samples=np.asarray(samples);base=np.asarray(base,dtype='float64');count=len(samples)
    if count<4 or count%2:raise ValueError('an even population of at least four IID members is required')
    if samples.shape[1:]!=base.shape:raise ValueError('base shape differs from a field member')
    if not np.isfinite(samples).all() or not np.isfinite(base).all():raise ValueError('inputs must be finite')
    half=count//2
    return base+(samples[:half].astype('float64')-samples[half:])/np.sqrt(2.)


def score(samples,truth,base,geometry,lock):
    from weather_controls_summary import monthly_record,fair_mean_mse,event_records
    from probability_joint_audit import fair_energy
    from tools.terrain_diffusion.native_weather_evaluation import fair_crps,point_metrics
    K,L,M,T,Q,H=samples.shape
    records={f'crps_h{h}':monthly_record(fair_crps(samples[...,h],truth[...,h]),1) for h in range(H)}
    inc=[fair_crps(samples[...,h+1]-samples[...,h],truth[...,h+1]-truth[...,h]) for h in range(H-1)]
    records['increment']=monthly_record(np.stack(inc,axis=-1),1)
    records.update(event_records(samples,truth,lock))
    for h in (0,6):
        records[f'fair_mean_mse_h{h}']=monthly_record(fair_mean_mse(samples[...,h],truth[...,h]),1)
        records[f'base_mse_h{h}']=monthly_record((base[...,h].astype('float64')-truth[...,h])**2,1)
    es=fair_energy(samples.reshape(K,L*M*T,Q*H),truth.reshape(L*M*T,Q*H)).reshape(L,M,T)
    records['energy']=monthly_record(es,1)
    i,j=np.triu_indices(Q,1);distance=np.linalg.norm((geometry[i,:2]-geometry[j,:2])*1000,axis=-1)
    for lo,hi in [(0,50),(50,150),(150,300),(300,np.inf)]:
        a=i[(distance>=lo)&(distance<hi)];b=j[(distance>=lo)&(distance<hi)]
        for h in (0,6):
            sums=np.zeros(M);counts=np.zeros(M,dtype='int64')
            for start in range(0,len(a),32):
                aa=a[start:start+32];bb=b[start:start+32]
                value=fair_crps(samples[...,aa,h]-samples[...,bb,h],truth[...,aa,h]-truth[...,bb,h])
                row=monthly_record(value,1);sums+=row['monthly_score_sum'];counts+=row['monthly_valid_count']
            key=f'pair_{lo}-{hi:g}_h{h}'
            records[key]=dict(value=float(sums.sum()/counts.sum()),monthly_score_sum=sums.tolist(),monthly_valid_count=counts.tolist(),pair_count=len(a))
    ordered=np.sort(samples,axis=0);valid=np.isfinite(truth)
    coverage=np.where(valid,(truth>=ordered[1])&(truth<=ordered[14]),np.nan)
    return dict(records=records,rank2_15_coverage=np.nanmean(coverage,axis=(0,1,2,3)).tolist(),
        rank2_15_continuous_IID_reference=13/17,negative_sample_fraction=float(np.mean(samples<0)),
        base_point_metrics=point_metrics(base,truth),sample_mean_point_metrics=point_metrics(np.mean(samples,axis=0),truth))


def main():
    from campaign import RESULTS,STOP,read
    from weather_controls_summary import validate_family
    from tools.terrain_diffusion.data import sha,write_json
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True);args=p.parse_args()
    lockpath=Path(__file__).with_name('independent_pair_centering_lock.json');lock=read(lockpath);seed=args.seed
    if lock['pairs']!=[[k,k+16] for k in range(16)] or lock['input_members']!=32 or lock['output_members']!=16:
        raise ValueError('locked member recipe changed')
    root=RESULTS/'20261126_independent_pair_centering'/f's{seed}';root.mkdir(parents=True,exist_ok=True)
    geometry=np.asarray(read(RESULTS/f'20261126_native_weather_cache_v2/calendar8_s{seed}/geometry.json')['dev_static'])
    baseline=None
    for label in lock['models']:
        if STOP.exists():raise SystemExit(75)
        solver=label.split('_')[1] if label!='reference' else None;mode=label.split('_')[0]
        folder=RESULTS/f'20261126_weather_projective_gp/s{seed}/c265/eval' if label=='reference' else RESULTS/f'20261126_weather_transport/s{seed}/{mode}/eval_{solver}'
        lineage=validate_family('transport',seed,mode,folder,solver);expected=lock['sources'][f's{seed}/{label}']
        if sha(folder/'lineage.json')!=expected['lineage_sha256'] or lineage['prediction_sha256']!=expected['prediction_sha256']:
            raise ValueError('frozen source differs from pairing lock')
        with np.load(folder/'evaluation.npz') as data:
            samples=data['samples'];truth=data['truth'];base=data['prediction']
        if baseline is None:baseline=(truth.copy(),base.copy())
        else:
            np.testing.assert_array_equal(truth,baseline[0]);np.testing.assert_array_equal(base,baseline[1])
        for variant in lock['families']:
            if STOP.exists():raise SystemExit(75)
            field=samples[:16].astype('float64') if variant=='raw_first16' else independent_pair_center(samples,base)
            result=score(field,truth,base,geometry,lock)
            payload=dict(seed=seed,model=label,variant=variant,input_members=32,members=16,
                **result,source=expected,source_model=lineage['source'],lock_sha256=sha(lockpath),code_sha256=sha(__file__),
                dependency_sha256={name:sha(Path(__file__).with_name(name)) for name in ['weather_controls_summary.py','probability_joint_audit.py','event_audit.py']},
                analytic_mean_is_base=variant!='raw_first16',covariance_claim='conditional population covariance, not realized finite-ensemble sample covariance',
                derived_arrays_saved=False,reproduction='frozen source NPZ plus locked disjoint member transform; no model fitting',
                strict_oof=False,confirmation_C_D_read=False)
            out=root/label/variant;out.mkdir(parents=True,exist_ok=True);write_json(out/'metrics.json',payload)
            print(json.dumps(dict(seed=seed,model=label,variant=variant,F6=result['records']['crps_h6']['value'],increment=result['records']['increment']['value'],energy=result['records']['energy']['value'])),flush=True)
            del field
        del samples


if __name__=='__main__':main()
