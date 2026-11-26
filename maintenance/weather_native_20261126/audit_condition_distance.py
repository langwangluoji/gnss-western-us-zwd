"""Reconstruct cached nearest distance from contemporaneous GNSS masks."""
import numpy as np
from campaign import RESULTS,read
from tools.terrain_diffusion.data import sha,write_json


def main():
    rows=[]
    for calendar in (8,15):
        for seed in (42,17):
            cache=RESULTS/f'20261126_native_weather_cache_v2/calendar{calendar}_s{seed}'
            manifest=read(cache/'manifest.json');geometry=read(cache/'geometry.json')
            if sha(cache/'geometry.json')!=manifest['geometry_sha256']:raise ValueError('geometry changed')
            arrays={}
            for name in ['history','context_ids','query_ids','nearest','origins']:
                if sha(cache/f'{name}.npy')!=manifest['files'][f'{name}.npy']:raise ValueError('array changed '+name)
                arrays[name]=np.load(cache/f'{name}.npy',mmap_mode='r')
            station=np.asarray(geometry['train_static']);query=np.asarray(geometry['dev_static'])
            mismatch=geometric_mismatch=mask_changed=0;max_error=0.;cells=0;max_geometric_difference=0.
            for i in range(len(arrays['history'])):
                h=arrays['history'][i];origin_count=len(arrays['origins'][i]);qid=arrays['query_ids'][i]
                if h.shape[0]!=72+origin_count-1:raise ValueError('unexpected unfolded history positions')
                distance=np.linalg.norm((query[qid,None,:2]-station[arrays['context_ids'][i]][None,:,:2])*1000,axis=-1)
                valid=h[-origin_count:,:,3]>0
                expected=np.where(valid[:,None],distance[None],np.inf).min(-1)
                geometric=distance.min(-1)[None];actual=arrays['nearest'][i]
                error=np.abs(expected-actual);geometric_error=np.abs(geometric-actual)
                if not np.isfinite(error).all():raise ValueError('nonfinite reconstructed distance')
                mismatch+=int((error>.001).sum());geometric_mismatch+=int((geometric_error>.001).sum())
                mask_changed+=int((np.abs(expected-geometric)>.001).sum());cells+=actual.size
                max_error=max(max_error,float(error.max()));max_geometric_difference=max(max_geometric_difference,float(geometric_error.max()))
            if mismatch:raise ValueError('nearest cache is not current-mask-aware')
            rows.append(dict(calendar_day=calendar,seed=seed,total_query_origin_cells=cells,
                current_gnss_mask_distance_mismatch_over_1m=mismatch,geometric_only_mismatch_over_1m=geometric_mismatch,
                mask_changes_nearest_cells=mask_changed,max_current_mask_difference_km=max_error,
                max_geometric_only_difference_km=max_geometric_difference,
                cache_manifest_sha256=sha(cache/'manifest.json'),geometry_sha256=manifest['geometry_sha256']))
    out=RESULTS/'20261126_condition_distance_audit';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'audit.json',dict(rows=rows,code_sha256=sha(__file__),tolerance_km=.001,
        conclusion='nearest.npy is distance to nearest context station with valid current GNSS, reconstructed from cached history mask; not geometry-only',
        no_training_or_label_role_change=True,confirmation_C_D_read=False))
    print(rows,flush=True)


if __name__=='__main__':main()
