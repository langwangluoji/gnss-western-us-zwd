"""Derive three already-built outage blocks without changing their values."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_weather_cache import static


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();source=args.source;out=args.out
    status=json.loads((source/'status.json').read_text(encoding='utf-8'))
    if status['completed']<1688:raise ValueError('outage prefix not yet complete')
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True,exist_ok=False)
    for path in source.glob('*.npy'):
        array=np.load(path,mmap_mode='r');np.save(out/path.name,np.array(array[1685:1688]))
    protocol['config']['out']=str(out);protocol['config']['blocks']=3
    protocol['derived_smoke']=dict(parent_protocol_sha256=sha(source/'protocol.json'),rows=[1685,1686,1687],
        method='exact unmodified contiguous row copy',script_sha256=sha(Path(__file__)))
    write_json(out/'protocol.json',protocol)
    geo=dict(np.load(Path(protocol['config']['data'])/'geometry.npz'))
    write_json(out/'geometry.json',dict(train_static=static(geo).tolist(),dev_static=static(geo,'dev_').tolist()))
    write_json(out/'manifest.json',dict(protocol_sha256=sha(out/'protocol.json'),geometry_sha256=sha(out/'geometry.json'),
        files={p.name:sha(p) for p in out.glob('*.npy')},complete_blocks=3))
    write_json(out/'status.json',dict(status='COMPLETE',completed=3,total=3))
    truth=np.load(out/'truth.npy');origins=np.load(out/'origins.npy')
    if not (np.diff(origins.ravel())==1).all():raise ValueError('derived smoke lost chronological order')
    print(json.dumps(dict(blocks=3,valid_trajectories_per_block=np.isfinite(truth).all(-1).sum((1,2)).tolist(),
        missing_label_cells=int(np.isnan(truth).sum()),target_values_unmodified=True)))


if __name__=='__main__':main()
