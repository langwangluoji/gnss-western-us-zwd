"""Query subset audit on actual trained p14 fusion weights, in physical mm."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .p14_fusion import FusionData,P14Fusion
from .data import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    data=FusionData(a.data,72)
    anchor=np.searchsorted(data.events.hours,np.datetime64('2020-01-08','h').astype('int64'))
    b=data.batch(np.arange(anchor,anchor+2),data.events.layouts[0],True,True,True)
    indices=np.random.default_rng(99).choice(137,64,replace=False)
    subset=dict(b)
    for key in ('point','static','time_uv'):subset[key]=b[key][:,indices]
    subset['uv']=b['uv'][:,:,indices]
    rows=[]
    for seed in (42,17):
        for kind in ('p14','features','mlp','gru','lstm','transformer'):
            path=a.out/f'{kind}_s{seed}/last.pt'
            if not path.exists():continue
            model=P14Fusion(kind,72).cuda().eval();model.load_state_dict(torch.load(path,weights_only=True))
            delta=((model(b)[:,indices]-model(subset))*data.std[0]).cpu().numpy()
            rows.append(dict(kind=kind,seed=seed,queries=[137,64],rms_drift_mm=float(np.sqrt(np.mean(delta**2))),
                max_drift_mm=float(np.abs(delta).max())))
            del model
    write_json(a.out/'query_consistency.json',rows)
    print(rows,flush=True)


if __name__=='__main__':main()
