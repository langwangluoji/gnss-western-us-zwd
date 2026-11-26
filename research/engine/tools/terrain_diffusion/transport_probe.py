"""Frozen pointwise query-expansion probe for v11 checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import Dataset,write_json,sha
from .transport import TransportModel
from .transport_train import ROOT
from .diagnose import subset


@torch.no_grad()
def run(a):
    torch.set_num_threads(4);data=Dataset();out=[]
    chosen=np.sort(np.random.default_rng(291).choice(137,a.size,False))
    for name in a.runs:
        path=ROOT/name/'best.pt';pack=torch.load(path,map_location='cuda',weights_only=False);conf=pack['config']
        model=TransportModel(conf['variant'],data.mean[0],data.std[0]).cuda().eval();model.load_state_dict(pack['model'])
        changes=[]
        for li,c in enumerate(data.layouts()):
            for row in (0,6):
                base=data.batch(row,12,c,np.arange(137),'validation');k=32 if conf['arm']!='deterministic' else 1
                d={key:v.expand(k,*v.shape[1:]) for key,v in base.items()}
                gen=torch.Generator(device='cuda').manual_seed(333+li*20+row)
                noise=torch.randn(k,137,device='cuda',generator=gen)
                if conf['arm']=='diffusion':
                    full=model.sample(d,noise);part=model.sample(subset(d,chosen),noise[:,chosen])
                else:
                    mean,scale=model.baseline(d);full=mean if conf['arm']=='deterministic' else mean+noise*scale
                    mean,scale=model.baseline(subset(d,chosen));part=mean if conf['arm']=='deterministic' else mean+noise[:,chosen]*scale
                changes.append(((part-full[:,chosen])*data.std[0]).cpu().numpy())
        delta=np.stack(changes);out.append(dict(run=name,query_size=a.size,checkpoint_sha256=sha(path),member_rms_change_mm=float(np.sqrt(np.mean(delta**2))),
             mean_rms_change_mm=float(np.sqrt(np.mean(delta.mean(1)**2))),note='Common-noise path sensitivity; not by itself a distributional consistency proof'))
    write_json(ROOT/('query_expansion.json' if a.size==32 else f'query_expansion{a.size}.json'),out);print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--size',type=int,default=32);run(p.parse_args())
