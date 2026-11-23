"""Actual residual backbone, dual optimizer, RNG and EMA exact GPU restart gate."""
import os
import copy
import tempfile
from pathlib import Path
import numpy as np
import torch
from field_generation_v1.model import ResidualField,gaussian_losses
from field_generation_v1.training import save_checkpoint,load_checkpoint,update_ema
from field_generation_v1.geometry import mosaic,query_grid
from field_generation_v1.prepare import ROOT
from residual_edm_v1.data import write_json

def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    def create():
        torch.manual_seed(42);m=ResidualField().cuda();e=copy.deepcopy(m);rng=np.random.default_rng(42)
        var=list(m.variance_head.parameters());ids={id(p) for p in var};mean=[p for p in m.parameters() if id(p) not in ids]
        opts=[torch.optim.AdamW(mean,lr=1e-4,weight_decay=1e-4),torch.optim.AdamW(var,lr=1e-4,weight_decay=1e-4)]
        return m,e,opts,rng
    def steps(m,e,opts,rng,start,stop):
        for step in range(start,stop):
            b=dict(stations=torch.randn(1,4,17,device='cuda'),padding=torch.zeros(1,4,dtype=torch.bool,device='cuda'),coarse=torch.randn(1,4,15,6,6,device='cuda'),maps=torch.randn(1,6,24,24,device='cuda'),baseline_condition=torch.zeros(1,1,24,24,device='cuda'),mu_condition=torch.zeros(1,1,24,24,device='cuda'),domain=torch.ones(1,1,24,24,device='cuda'))
            for o in opts:
                for g in o.param_groups:g['lr']=1e-4*(step+1)/500
                o.zero_grad(set_to_none=True)
            r,f=m(b);r=mosaic(r,np.array([[0,0]]),(24,24));f=mosaic(f,np.array([[0,0]]),(24,24))
            xy=torch.tensor([[12.3,11.8]],device='cuda');mean=query_grid(r,xy)[:,0]
            std=m.standard_deviation(query_grid(f,xy),mean.new_zeros(1,3),1,True)
            mse,nll=gaussian_losses(mean,mean.new_tensor([rng.normal()]),std.square(),1,True)
            (mse+nll).backward()
            for o in opts:torch.nn.utils.clip_grad_norm_(o.param_groups[0]['params'],1);o.step()
            update_ema(e,m)
    m,e,o,r=create();steps(m,e,o,r,0,200);expected={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};expected_ema={k:v.detach().cpu().clone() for k,v in e.state_dict().items()}
    del m,e,o
    m,e,o,r=create();steps(m,e,o,r,0,100)
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        path=Path(temp)/'resume.pt';save_checkpoint(path,m,e,o,r,100,{'cursor':100});del m,e,o
        m,e,o,r=create();saved=load_checkpoint(path,m,e,o,r);steps(m,e,o,r,100,200)
    for key,value in m.state_dict().items():torch.testing.assert_close(value.cpu(),expected[key],rtol=0,atol=0)
    for key,value in e.state_dict().items():torch.testing.assert_close(value.cpu(),expected_ema[key],rtol=0,atol=0)
    write_json(ROOT/'restore_gate.json',dict(passed=True,updates=200,resume_at=100,exact=True,model='ResidualField',optimizers=2,device='cuda'))
    print('EXACT REAL MODEL RESTORE PASS',flush=True)

if __name__=='__main__':main()
