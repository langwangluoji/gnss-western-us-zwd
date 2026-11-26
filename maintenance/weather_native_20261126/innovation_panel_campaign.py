"""Fixed exact-month leave-out information panel; no new neural training."""
import argparse
from pathlib import Path
import time
import numpy as np
from campaign import RESULTS,read,run,STOP
from tools.terrain_diffusion.native_innovation_audit import codes
from tools.terrain_diffusion.data import sha,write_json


def schedule(origins):
    rows=[]
    for year in range(2016,2020):
        for month in range(1,13):
            hour=int(np.datetime64(f'{year}-{month:02d}-08T01','h').astype('int64'))
            found=np.flatnonzero(origins[:,0]==hour)
            if not len(found):raise ValueError('fixed audit hour absent from cache')
            i=int(found[0])
            if not np.array_equal(origins[i],hour+np.arange(12)):raise ValueError('panel block not12 consecutive hours')
            rows.append(dict(year=year,month=month,index=i,first_hour=hour))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True);args=p.parse_args();seed=args.seed
    cache=RESULTS/f'20261126_native_weather_cache_v2/train_s{seed}'
    embedding=RESULTS/f'20261126_native_weather_embeddings/transformer72_physics_aux_last_train_s{seed}'
    origins=np.load(cache/'origins.npy',mmap_mode='r');rows=schedule(origins)
    root=RESULTS/f'20261126_leaveout_innovation_panel/s{seed}';root.mkdir(parents=True,exist_ok=True)
    protocol=dict(seed=seed,schedule=rows,origins_per_block=12,code_sha256=codes(),campaign_sha256=sha(__file__),
        spec_sha256=sha(Path(__file__).resolve().parents[2]/'docs/superpowers/specs/2026-11-26-leaveout-innovation-panel.md'),
        cache_manifest_sha256=sha(cache/'manifest.json'),mean_manifest_sha256=sha(embedding/'manifest.json'),
        role='2016-18 probe fit;2019 probe transfer; background already fitted2016-19; exposed training-pool diagnostic',
        strict_oof=False,confirmation_C_D_read=False)
    if (root/'protocol.json').exists():
        if read(root/'protocol.json')!=protocol:raise ValueError('panel protocol changed')
    else:write_json(root/'protocol.json',protocol)
    start=time.monotonic()
    for number,row in enumerate(rows):
        if STOP.exists():raise SystemExit(75)
        out=RESULTS/f"20261126_leaveout_innovation_audit/b{row['index']:04d}_o12_s{seed}"
        run('native_innovation_audit',['--cache',cache,'--embedding',embedding,'--out',out,
            '--index',row['index'],'--origins',12],out/'summary.json')
        summary=read(out/'summary.json')
        if not summary['complete'] or summary['code_sha256']!=codes() or summary['arrays_sha256']!=sha(out/'audit_arrays.npz'):
            raise ValueError('panel block incomplete or changed')
        if summary['cache_manifest_sha256']!=protocol['cache_manifest_sha256'] or summary['mean_manifest_sha256']!=protocol['mean_manifest_sha256']:
            raise ValueError('panel block source differs')
        write_json(root/'progress.json',dict(completed=number+1,total=len(rows),row=row,seconds=time.monotonic()-start))
        print('PANEL_COMPLETE_BLOCK',seed,number+1,len(rows),flush=True)
    write_json(root/'status.json',dict(status='COMPLETE_INFORMATION_PANEL',blocks=len(rows),seconds=time.monotonic()-start,
        protocol_sha256=sha(root/'protocol.json')))


if __name__=='__main__':main()
