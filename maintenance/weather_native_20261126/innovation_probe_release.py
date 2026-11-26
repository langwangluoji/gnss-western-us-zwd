"""Seal2016-2018 probe coefficients before scoring any2019 panel labels."""
import argparse
from pathlib import Path
import numpy as np
from campaign import RESULTS,read
from innovation_panel_summary import fit_probe,transfer_features
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_innovation_audit import codes


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True);args=p.parse_args();seed=args.seed
    root=RESULTS/f'20261126_leaveout_innovation_panel/s{seed}';protocol=read(root/'protocol.json')
    lock_path=Path(__file__).with_name('innovation_panel_lock.json');lock=read(lock_path)
    for name,value in lock['new_sources'].items():
        if sha(Path(__file__).with_name(name))!=value:raise ValueError('frozen probe code changed')
    if protocol['code_sha256']!=codes() or sha(root/'protocol.json')!=lock['panel_protocol_sha256'][str(seed)]:raise ValueError('panel source changed')
    selected=[r for r in protocol['schedule'] if r['year'] in (2016,2017,2018)]
    if len(selected)!=36:raise ValueError('requires36 training-probe months')
    cache=RESULTS/f'20261126_native_weather_cache_v2/train_s{seed}'
    weight=np.load(cache/'weights.npy',mmap_mode='r');target=[];weights=[];years=[];fields={n:[] for n in ['self','LOSO','LOGO']};sources={}
    for row in selected:
        i=row['index'];block=RESULTS/f'20261126_leaveout_innovation_audit/b{i:04d}_o12_s{seed}'
        summary=read(block/'summary.json')
        if not summary['complete'] or summary['code_sha256']!=codes() or summary['arrays_sha256']!=sha(block/'audit_arrays.npz'):
            raise ValueError('training-probe block incomplete or changed')
        if summary['cache_manifest_sha256']!=protocol['cache_manifest_sha256']:raise ValueError('training-probe cache differs')
        with np.load(block/'audit_arrays.npz') as a:
            for arm in fields:fields[arm].append(transfer_features(a[arm+'_error'],a['query_xy'],a['context_xy'],a['query_height'],a['context_height']).reshape(-1,9))
            target.append((a['query_truth'].astype('float64')-a['query_mean'].astype('float64')).reshape(-1,7))
        weights.append(np.asarray(weight[i]).reshape(-1));years.append(row['year']);sources[str(i)]=sha(block/'summary.json')
    fields={n:np.stack(x) for n,x in fields.items()};target=np.stack(target);weights=np.stack(weights);years=np.asarray(years)
    fields['intercept']=np.empty((*target.shape[:-1],0))
    fields['self_LOSO']=np.concatenate([fields['self'],fields['LOSO']],-1);fields['self_LOGO']=np.concatenate([fields['self'],fields['LOGO']],-1)
    fits={n:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fit_probe(x,target,weights,years).items()} for n,x in fields.items()}
    result=dict(seed=seed,fits=fits,training_years=[2016,2017,2018],blocks=36,source_summary_sha256=sources,
        frozen_probe_lock_sha256=sha(lock_path),code_sha256=sha(__file__),panel_protocol_sha256=sha(root/'protocol.json'),
        uses2019_labels_or_features=False,source_foundation_already_fitted2019=True,strict_oof=False,confirmation_C_D_read=False)
    out=root/'probe_fit_release.json'
    if out.exists():
        if read(out)!=result:raise ValueError('released probe fit changed')
    else:write_json(out,result)
    print('Released36-month fits',seed,list(fits),flush=True)


if __name__=='__main__':main()
