"""Verify reconstructed self kernels/lags against cached condition first9."""
from pathlib import Path
import numpy as np
from campaign import RESULTS,read
from innovation_panel_summary import transfer_features
from tools.terrain_diffusion.data import sha,write_json


def main():
    rows=[]
    for seed in (42,17):
        root=RESULTS/f'20261126_leaveout_innovation_panel/s{seed}';protocol=read(root/'protocol.json')
        if read(root/'status.json')['status']!='COMPLETE_INFORMATION_PANEL':raise ValueError('panel incomplete')
        cache=RESULTS/f'20261126_native_weather_cache_v2/train_s{seed}';cm=read(cache/'manifest.json')
        if sha(cache/'condition.npy')!=cm['files']['condition.npy']:raise ValueError('condition cache changed')
        condition=np.load(cache/'condition.npy',mmap_mode='r')
        for row in protocol['schedule']:
            i=row['index'];folder=RESULTS/f'20261126_leaveout_innovation_audit/b{i:04d}_o12_s{seed}';summary=read(folder/'summary.json')
            if sha(folder/'audit_arrays.npz')!=summary['arrays_sha256']:raise ValueError('audit arrays changed')
            with np.load(folder/'audit_arrays.npz') as a:
                features=transfer_features(a['self_error'],a['query_xy'],a['context_xy'],a['query_height'],a['context_height'])
            cached=np.asarray(condition[i,...,:9],dtype='float32');difference=float(np.max(np.abs(features-cached)))
            if difference>.005:raise ValueError('self reconstructed feature differs from cache')
            rows.append(dict(seed=seed,**row,max_absolute_feature_difference=difference,arrays_sha256=summary['arrays_sha256']))
    out=RESULTS/'20261126_leaveout_innovation_panel/paired';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'feature_equivalence.json',dict(rows=rows,tolerance=.005,max_absolute_difference=max(r['max_absolute_feature_difference'] for r in rows),
        conclusion='same9 self innovation fields: lag0/1/3 times50/150/300km; float16-cache tolerance',
        code_sha256=sha(__file__),transfer_source_sha256=sha(Path(__file__).with_name('innovation_panel_summary.py')),
        strict_oof=False,confirmation_C_D_read=False))
    print('Reconstructed self features verified:',len(rows),'blocks; max difference',max(r['max_absolute_feature_difference'] for r in rows),flush=True)


if __name__=='__main__':main()
