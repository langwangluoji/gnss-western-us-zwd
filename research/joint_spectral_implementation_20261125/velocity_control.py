"""Same data/representation/seed/budget: velocity vs archived epsilon100 pilot."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.joint_experiment import load_events,probability_run
from tools.terrain_diffusion.velocity_diffusion import VelocityDiffusion
from tools.terrain_diffusion.data import sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    source=args.run;out=source/'velocity_control';out.mkdir(exist_ok=False)
    (out/'representation').mkdir()
    for f in (source/'representation').iterdir():
        if f.is_file():shutil.copy2(f,out/'representation'/f.name)
    torch.set_num_threads(4);train=load_events(source/'oof','train');inner=load_events(source/'oof','inner')
    options=SimpleNamespace(seed=42,diagnostic=True,probability_steps=100,members=16,ddim_steps=20)
    write_json(out/'protocol.json',dict(source_run=str(source),source_protocol_sha256=sha(source/'protocol_all.json'),
        source_code_sha256=json.loads((source/'protocol_all.json').read_text())['code_sha256'],
        design='post-pilot numerical stability comparison, same100-step budget; not a formal skill ranking',
        args=vars(options),runner_sha256=sha(__file__),code_sha256={f:sha(ENGINE/'tools/terrain_diffusion'/f) for f in ('joint_experiment.py','joint_probability.py','velocity_diffusion.py')},
        source_oof_hashes={str(f.relative_to(source)):sha(f) for f in (source/'oof').glob('fold*/*_receipt.json')}))
    result={}
    for arm in ('A0','A2','B1','J1'):
        result[arm]=probability_run(out,train,inner,options,arm,diffusion_class=VelocityDiffusion,cache_root=source)
        write_json(out/'summary.json',result)
    write_json(out/'status.json',dict(status='COMPLETE_4_VELOCITY_CONTROLS',diagnostic=True))


if __name__=='__main__':main()
