"""Frozen lead-dependent mean interventions with unchanged random residual fields."""
import argparse
import json
from pathlib import Path
import numpy as np


def mean_centers(base,gaussian):
    base=np.asarray(base,dtype='float64');gaussian=np.asarray(gaussian,dtype='float64')
    if base.shape!=gaussian.shape:raise ValueError('mean source shape differs')
    if base.shape[-1]!=7:raise ValueError('exactly seven registered leads required')
    if not np.isfinite(base).all() or not np.isfinite(gaussian).all():raise ValueError('mean sources must be finite')
    hybrid=gaussian.copy();hybrid[...,0]=base[...,0]
    return dict(base=base.copy(),gaussian=gaussian.copy(),hybrid=hybrid)


def move_center(samples,old,new):
    samples=np.asarray(samples,dtype='float64');old=np.asarray(old,dtype='float64');new=np.asarray(new,dtype='float64')
    if samples.shape[1:]!=old.shape or old.shape!=new.shape:raise ValueError('field/center shape differs')
    if not np.isfinite(samples).all() or not np.isfinite(old).all() or not np.isfinite(new).all():raise ValueError('field and centers must be finite')
    result=samples+(new-old)
    if not np.isfinite(result).all():raise ValueError('nonfinite moved field')
    return result


def verified_geometry(cache):
    import hashlib
    path=cache/'geometry.json';fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    if fingerprint!=manifest.get('geometry_sha256'):raise ValueError('geometry differs from frozen cache manifest')
    geometry=np.asarray(json.loads(path.read_text(encoding='utf-8'))['dev_static'])
    if geometry.ndim!=2 or geometry.shape[0]!=137 or geometry.shape[1]<2 or not np.isfinite(geometry).all():
        raise ValueError('geometry is not the registered finite137-query population')
    return geometry,fingerprint


def score_intervention(samples,truth,base,center,geometry,lock,std=None):
    from independent_pair_centering import score
    from weather_controls_summary import monthly_record
    from tools.terrain_diffusion.native_weather_evaluation import gaussian_crps,point_metrics
    result=score(samples,truth,base,geometry,lock)
    # The inherited scorer's fair scores support any K; replace its K16-only
    # coverage display with the actual registered member count here.
    result.pop('rank2_15_coverage');result.pop('rank2_15_continuous_IID_reference')
    K=len(samples);lo,hi=(2,15) if K==16 else (3,30) if K==32 else (None,None)
    if lo is None:raise ValueError('unregistered member population')
    ordered=np.sort(samples,axis=0)
    coverage=np.where(np.isfinite(truth),(truth>=ordered[lo-1])&(truth<=ordered[hi-1]),np.nan)
    result.update(rank_interval=[lo,hi],rank_interval_continuous_IID_reference=(hi-lo)/(K+1),
        rank_interval_coverage=np.nanmean(coverage,axis=(0,1,2,3)).tolist(),analytic_center_point_metrics=point_metrics(center,truth))
    for h in (0,6):
        result['records'][f'analytic_center_mse_h{h}']=monthly_record((center[...,h]-truth[...,h])**2,1)
    if std is not None:
        for h in range(7):result['records'][f'analytic_crps_h{h}']=monthly_record(gaussian_crps(center[...,h],std[...,h],truth[...,h]),1)
    return result


def checked_source(seed,day,label,lock):
    from campaign import RESULTS
    from weather_controls_summary import validate_family
    from tools.terrain_diffusion.data import sha
    if label=='reference':
        folder=RESULTS/f'20261126_weather_projective_gp/s{seed}/c265/eval' if day==8 else RESULTS/f'20261126_weather_gp_calendar15_cpu/s{seed}/c265'
        lineage=validate_family('transport',seed,'reference',folder) if day==8 else validate_family('supplement',seed,'c265',folder)
    else:
        mode,solver=label.split('_');folder=RESULTS/f'20261126_weather_transport/s{seed}/{mode}/eval_{solver}'
        lineage=validate_family('transport',seed,mode,folder,solver)
    expected=lock['sources'][f'calendar{day}/s{seed}/{label}']
    if sha(folder/'lineage.json')!=expected['lineage_sha256'] or lineage['prediction_sha256']!=expected['prediction_sha256']:
        raise ValueError('frozen center-intervention source changed')
    return folder,lineage


def main():
    from campaign import RESULTS,STOP,read,ENGINE
    from independent_pair_centering import independent_pair_center
    from tools.terrain_diffusion.data import sha,write_json
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True)
    p.add_argument('--phase',choices=['gaussian','paired'],required=True);args=p.parse_args();seed=args.seed
    lockpath=Path(__file__).with_name('lead_mean_audit_lock.json');lock=read(lockpath)
    if lock['centers']!=['base','gaussian','hybrid'] or lock['pairs']!=[[k,k+16] for k in range(16)]:raise ValueError('frozen center/pair recipe changed')
    root=RESULTS/'20261126_lead_mean_audit_v2'/args.phase/f's{seed}'
    dependencies={n:sha(Path(__file__).with_name(n)) for n in ['independent_pair_centering.py','weather_controls_summary.py','probability_joint_audit.py','event_audit.py']}
    dependencies['native_weather_evaluation.py']=sha(ENGINE/'tools/terrain_diffusion/native_weather_evaluation.py')
    days=lock['gaussian_calendars'] if args.phase=='gaussian' else [lock['paired_calendar']]
    for day in days:
        if STOP.exists():raise SystemExit(75)
        parent,parent_lineage=checked_source(seed,day,'reference',lock)
        with np.load(parent/'evaluation.npz') as data:
            base=data['prediction'];truth=data['truth'];mu=data['gaussian_distribution_mean'];std=data['gaussian_distribution_std']
        if not np.isfinite(std).all() or np.any(std<=0):raise ValueError('Gaussian scale invalid')
        centers=mean_centers(base,mu)
        cache=RESULTS/f'20261126_native_weather_cache_v2/calendar{day}_s{seed}'
        geometry,geometry_sha=verified_geometry(cache)
        models=['reference'] if args.phase=='gaussian' else lock['paired_models']
        for label in models:
            folder,lineage=checked_source(seed,day,label,lock)
            with np.load(folder/'evaluation.npz') as data:
                samples=data['samples'];np.testing.assert_array_equal(data['truth'],truth);np.testing.assert_array_equal(data['prediction'],base)
            if samples.shape!=(32,3,12,72,137,7):raise ValueError('unregistered source population')
            residual=independent_pair_center(samples,np.zeros_like(base)) if args.phase=='paired' else None
            for name,center in centers.items():
                if STOP.exists():raise SystemExit(75)
                field=center+residual if args.phase=='paired' else move_center(samples,mu,center)
                if args.phase=='gaussian' and name=='gaussian':np.testing.assert_array_equal(field,samples)
                # A pure center intervention keeps every member's random part.
                expected=residual if args.phase=='paired' else samples.astype('float64')-mu
                np.testing.assert_allclose(field-center,expected,atol=6e-14,rtol=0)
                result=score_intervention(field,truth,base,center,geometry,lock,std.astype('float64') if args.phase=='gaussian' else None)
                payload=dict(seed=seed,phase=args.phase,calendar=day,model=label,center=name,members=len(field),**result,
                    source=lock['sources'][f'calendar{day}/s{seed}/{label}'],center_source=lock['sources'][f'calendar{day}/s{seed}/reference'],
                    source_model=lineage['source'],parent_model=parent_lineage['source'],analytic_center_known=True,
                    exact_unchanged_residual_verified=True,lock_sha256=sha(lockpath),code_sha256=sha(__file__),dependency_sha256=dependencies,
                    geometry_sha256=geometry_sha,cache_manifest_sha256=sha(cache/'manifest.json'),
                    derived_arrays_saved=False,strict_oof=False,confirmation_C_D_read=False)
                out=root/f'calendar{day}'/label/name;out.mkdir(parents=True,exist_ok=True);write_json(out/'metrics.json',payload)
                print(json.dumps(dict(seed=seed,phase=args.phase,calendar=day,model=label,center=name,
                    R=result['records']['crps_h0']['value'],F6=result['records']['crps_h6']['value'],increment=result['records']['increment']['value'])),flush=True)
                del field
            del samples,residual


if __name__=='__main__':main()
