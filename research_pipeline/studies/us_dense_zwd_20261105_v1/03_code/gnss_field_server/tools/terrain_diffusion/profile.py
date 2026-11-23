"""Real-data full four-hour, four-member, DDIM20 backward feasibility."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from .data import Dataset,write_json
from .model import DiffusionModel
from .train import trajectory_loss,seed_all


def run(out):
    torch.set_num_threads(4);seed_all(42);data=Dataset();c=data.layouts()[0]
    q=np.flatnonzero(~np.isin(data.a['groups'],data.a['groups'][c])&np.isfinite(data.a['train'][0,12:16,:,0]).all(0))[:16]
    batches=[data.batch(0,t,c,q) for t in range(12,16)]
    model=DiffusionModel().cuda().train();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    delta=np.diff(data.a['train'][...,0],axis=1);ratio=float(np.nanstd(delta)/data.std[0])
    loss,seq=trajectory_loss(model,batches,4,20,ratio);loss.backward();torch.cuda.synchronize()
    result=dict(status='complete',hours=4,members=4,ddim_steps=20,queries=len(q),context=len(c),
                seconds=time.perf_counter()-start,peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,
                loss=float(loss.detach()),finite_gradients=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
                history_gradient='full graph; no detach except auxiliary v preservation',increment_scale=ratio)
    write_json(out,result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(Path(p.parse_args().out))
