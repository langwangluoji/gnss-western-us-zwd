"""Independent-noise validation or joint inference at arbitrary surface queries."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import Dataset,static_geometry,write_json,sha
from .model import DiffusionModel
from .train import validate,history_push


def run(a):
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    pack=torch.load(a.checkpoint,map_location='cuda',weights_only=False);config=pack['config']
    data=Dataset(config['data'])
    if data.contract['data_sha256']!=config['data_sha256']:raise ValueError('Checkpoint data contract mismatch')
    model=DiffusionModel(config['width'],config['blocks']).cuda();model.load_state_dict(pack['model']);model.eval()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if a.query_csv:
        frame=pd.read_csv(a.query_csv)
        if not {'x','y','height'}<=set(frame):raise ValueError('Query CSV needs projected x,y (metres), height (metres), in source geometry CRS')
        geometry=static_geometry(frame);c=data.layouts()[a.layout];q=np.arange(len(frame));history=None;seq=[]
        with torch.no_grad():
            for t in range(12,12+a.hours):
                d=data.batch(a.row,t,c,q,'validation',query_geometry=geometry)
                d={k:v.expand(a.members,*v.shape[1:]) for k,v in d.items()}
                if history is not None:d['history']=history
                noise=torch.randn_like(d['target'])
                if config['arm']=='diffusion':p=model.sample(d,noise,a.ddim_steps)
                else:
                    mean,scale=model.baseline(d);p=mean+scale*noise if config['arm']=='gaussian' else mean
                history=history_push(d,p);seq.append((p*data.std[0]+data.mean[0]).cpu().numpy())
        np.savez_compressed(out/'field.npz',zwd_mm=np.stack(seq,1),x=frame.x,y=frame.y,height=frame.height)
        result=dict(status='joint_query_generation',queries=len(q),query_sha256=sha(a.query_csv),source_row=a.row,layout=a.layout)
    else:
        with torch.no_grad():result,samples,truth=validate(model,data,config['arm'],rows=tuple(a.rows),hours=a.hours,members=a.members,steps=a.ddim_steps,seed=a.seed)
        np.savez_compressed(out/'evaluation.npz',samples=samples,truth=truth)
    write_json(out/'metrics.json',result)
    write_json(out/'config.json',dict(checkpoint_sha256=sha(a.checkpoint),data=config['data'],data_sha256=data.contract['data_sha256'],**vars(a)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--hours',type=int,choices=range(1,25),default=24);p.add_argument('--members',type=int,default=16)
    p.add_argument('--ddim-steps',type=int,default=20);p.add_argument('--seed',type=int,default=19007);p.add_argument('--rows',type=int,nargs='+',default=[0,6])
    p.add_argument('--query-csv');p.add_argument('--row',type=int,default=0);p.add_argument('--layout',type=int,choices=[0,1,2],default=0)
    run(p.parse_args())
