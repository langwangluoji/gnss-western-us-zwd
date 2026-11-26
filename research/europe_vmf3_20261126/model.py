"""Zero-initialized background-feature correction on a frozen strong field."""
import argparse
import copy
import json
import time

import numpy as np
import torch
from torch import nn

from .features import HERE,CACHE,PARENT,OLD_CACHE,sha,write_json


class BackgroundCorrection(nn.Module):
    def __init__(self,use_vmf3=True):
        super().__init__();self.use_vmf3=use_vmf3
        self.time=nn.GRU(4,32,1,batch_first=True)
        self.head=nn.Sequential(nn.Linear(111+4+32,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,field,condition,history):
        if not self.use_vmf3:history=torch.zeros_like(history)
        B,L,Q,_=history.shape
        state=self.time(history.permute(0,2,1,3).reshape(B*Q,L,4))[0][:,-1].reshape(B,Q,32)
        value=torch.cat([condition,history[:,-1],state],-1)
        return field+10*self.head(value).squeeze(-1)


def batch(cache,positions,device='cuda'):
    p=np.asarray(positions)
    if p.min()<15 or not np.all(np.diff(p)==1):raise ValueError('requires consecutive origins and16past records')
    index=p[:,None]-np.arange(15,-1,-1)[None]
    return tuple(torch.tensor(v,device=device) for v in (cache['field'][p],cache['condition'][p],cache['vmf'][index]))


@torch.no_grad()
def monitor(model,cache,a):
    q=cache['query'];origins=np.linspace(int(cache['fit_start'])+16,len(cache['field'])-8,24).astype(int)
    square=np.zeros(len(q));count=np.zeros(len(q));model.eval()
    for p in origins:
        pos=np.arange(p,p+4);prediction=model(*batch(cache,pos)).cpu().numpy()
        target=a['values'][pos[:,None]+int(cache['start']),q[None],0];valid=np.isfinite(target)
        square+=np.where(valid,(prediction-target)**2,0).sum(0);count+=valid.sum(0)
    active=count>0
    return float(np.sqrt(square[active]/count[active]).mean())


def train(seed,arm,steps=3000,smoke=False):
    torch.set_num_threads(4);torch.manual_seed(seed);a=dict(np.load(OLD_CACHE/'data.npz'));root=CACHE/f's{seed}'
    caches=[dict(np.load(p)) for p in sorted(root.glob('train_*.npz'))];dev=dict(np.load(root/'dev.npz'))
    blocks=[(i,p) for i,c in enumerate(caches) for p in range(int(c['fit_start']),len(c['field'])-3,4)]
    model=BackgroundCorrection(arm=='vmf3').cuda();ema=copy.deepcopy(model);ema.time.flatten_parameters()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    out=HERE/('smoke' if smoke else 'runs')/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=False)
    initial=monitor(model,dev,a);best=initial;torch.save(model.state_dict(),out/'best.pt');start=time.monotonic()
    write_json(out/'lock.json',dict(seed=seed,arm=arm,steps=steps,lr=3e-4,initial_dev_rmse_mm=initial,initial_state='exact frozen I1 output',parent_selection_sha256=sha(PARENT/'selection_2023_v2.json'),feature_receipt_sha256=sha(HERE/f'features_s{seed}.json'),code_sha256=sha(__file__),order='consecutive four-record blocks over2020-2023, loop whole passes',blocks_per_pass=len(blocks),parameters=sum(p.numel() for p in model.parameters()),feature_multiplicative_gate=arm=='vmf3',target_station_GNSS_or_weather_condition=False))
    for step in range(1,steps+1):
        i,p=blocks[(step-1)%len(blocks)];c=caches[i];pos=np.arange(p,p+4);q=c['query']
        target=torch.tensor(a['values'][pos[:,None]+int(c['start']),q[None],0],device='cuda');valid=torch.isfinite(target)
        model.train();opt.zero_grad(set_to_none=True);prediction=model(*batch(c,pos));loss=((prediction[valid]-target[valid])/10).square().mean()
        loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise FloatingPointError('VMF3 correction nonfinite')
        opt.step()
        with torch.no_grad():
            for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,.01)
        if step==1 or step%100==0 or step==steps:
            row=dict(step=step,seconds=time.monotonic()-start,batch_rmse_mm=float(loss.detach().sqrt()*10),global_origin=int(c['start'])+p,gradient_norm=float(grad))
            if step%500==0 or step==steps:
                score=monitor(ema,dev,a);row['dev_station_macro_rmse_mm']=score
                if score<best:best=score;torch.save(ema.state_dict(),out/'best.pt')
            write_json(out/'progress.json',row)
            with (out/'curve.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(stage=out.name,**row)),flush=True)
    torch.save(model.state_dict(),out/'last.pt');torch.save(ema.state_dict(),out/'ema.pt')
    write_json(out/'complete.json',dict(steps=steps,seconds=time.monotonic()-start,best_dev_rmse_mm=best,best_sha256=sha(out/'best.pt')))


@torch.no_grad()
def predict(seed,arm):
    torch.set_num_threads(4);root=HERE/f'runs/{arm}_s{seed}';model=BackgroundCorrection(arm=='vmf3').cuda();model.load_state_dict(torch.load(root/'best.pt',weights_only=True));model.eval();model.time.flatten_parameters()
    cache=dict(np.load(CACHE/f's{seed}/test.npz'));first=int(cache['fit_start']);result=[]
    for p in range(first,len(cache['field']),16):result.append(model(*batch(cache,np.arange(p,min(p+16,len(cache['field']))))).cpu().numpy())
    a=np.load(OLD_CACHE/'data.npz');path=CACHE/f'{arm}_predictions_s{seed}.npz'
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,pred=np.concatenate(result),query=cache['query'],hours=a['hours'][np.arange(first,len(cache['field']))+int(cache['start'])])
    write_json(root/'evaluation_receipt.json',dict(weights_sha256=sha(root/'best.pt'),predictions_sha256=sha(path),selection_sha256=sha(HERE/'selection_2023.json'),epochs=len(np.concatenate(result))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--arm',choices=['control','vmf3'],default='vmf3');p.add_argument('--steps',type=int,default=3000);p.add_argument('--smoke',action='store_true');p.add_argument('--predict',action='store_true');args=p.parse_args()
    if args.predict:predict(args.seed,args.arm)
    else:train(args.seed,args.arm,args.steps,args.smoke)
