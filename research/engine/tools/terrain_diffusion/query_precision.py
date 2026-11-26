"""Continuous query-local temporal residuals on the exact ~10mm reference."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import PROJECT,sha,write_json
from .warm_joint_experiment import WarmData,Reference
from .masked_transformer_fusion import CausalSpatialEncoder
from .p14_fusion import chronological_blocks
from .p14_fusion_experiment import evaluate


class QueryResidual(nn.Module):
    def __init__(self,state,source_kind='gru',length=72,side=16,width=64):
        super().__init__();self.kind='Q';self.reference=Reference(state,source_kind,length)
        for p in self.reference.parameters():p.requires_grad_(False)
        self.encoder=CausalSpatialEncoder(width=width,side=side)
        self.norm=nn.LayerNorm(width)
        self.local=nn.Sequential(nn.Linear(123+width,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)

    def forward(self,b):
        reference=self.reference(b);h=self.encoder(b['history'],b['time_static'])
        q=F.grid_sample(h,b['time_uv'][:,None],padding_mode='border',align_corners=True)[:,:,0].transpose(1,2)
        f=torch.cat([b['point'],b['static'],self.norm(q),reference[...,None]],-1)
        raw=self.local(f).squeeze(-1)
        # time_scale=100/std_old: 5mm becomes .05*time_scale normalized units.
        limit=.05*b.get('time_scale',1.)
        return reference+limit*torch.tanh(raw/limit)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=PROJECT/'results/20261125_temporal_continuous_data')
    p.add_argument('--source',type=Path,default=PROJECT/'results/20261125_p14_continuous_fusion')
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--seeds',type=int,nargs='+',default=[42,17])
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--cuda-fraction',type=float,default=.38)
    p.add_argument('--resume',type=Path);args=p.parse_args()
    if args.resume is not None and len(args.seeds)!=1:raise ValueError('resume one checkpoint seed in a new output directory')
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(args.cuda_fraction)
    args.out.mkdir(parents=True,exist_ok=True);out=args.out
    protocol=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        code_sha256={f:sha(Path(__file__).with_name(f)) for f in ('query_precision.py','warm_joint_experiment.py','joint_data.py','masked_transformer_fusion.py','p14_fusion.py')},
        source_weights={str(s):sha(args.source/f'gru_s{s}/last.pt') for s in args.seeds},
        source_kind='gru',data_sha256=sha(args.data/'contract.json'),training='2016-2019 consecutive12h blocks, condition IDs only sampled',
        evaluation='2020 same12months x72h x3layouts x137dev; exposed development cohort',
        trust_region_mm=5.,ema_decay=.999,checkpoint_selection='final3000raw and final3000EMA preregistered, no earlier2020 checkpoint choice',
        strict_oof=False,confirmation_C_D_read=False)
    receipt=out/'protocol.json'
    if receipt.exists():
        if json.loads(receipt.read_text(encoding='utf-8'))!=protocol:raise ValueError('query run lineage differs')
    else:write_json(receipt,protocol)
    data=WarmData(args.data);e=data.events;stop=int(np.searchsorted(e.hours,np.datetime64('2020-01-01','h').astype('int64')))
    blocks=chronological_blocks(71,stop,12);summary=[]
    for seed in args.seeds:
        torch.manual_seed(seed);rng=np.random.default_rng(seed+28011)
        source=torch.load(args.source/f'gru_s{seed}/last.pt',weights_only=True)
        model=QueryResidual(source).cuda();ema=copy.deepcopy(model).eval()
        for module in ema.modules():
            if isinstance(module,nn.RNNBase):module.flatten_parameters()
        folder=out/f'trajectory_s{seed}';folder.mkdir(exist_ok=False)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
        trace=[];start_step=1;elapsed=0.;start=time.monotonic()
        if args.resume is not None:
            saved=torch.load(args.resume,weights_only=True)
            if saved['seed']!=seed or saved['lr']!=args.lr or saved['data_sha256']!=sha(args.data/'contract.json') or saved['source_sha256']!=sha(args.source/f'gru_s{seed}/last.pt'):
                raise ValueError('resume seed/lr/data/source mismatch')
            if saved['code_sha256']!=protocol['code_sha256']:raise ValueError('resume implementation differs')
            model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);opt.load_state_dict(saved['optimizer'])
            rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            start_step=saved['step']+1;trace=saved['trace'];elapsed=trace[-1]['seconds'] if trace else 0
        for step in range(start_step,args.steps+1):
            anchors=blocks[(step-1)%len(blocks)];c=np.sort(rng.choice(410,82,False));b=data.batch(anchors,c)
            pred=model(b);mask=b['mask'];loss=(pred-b['target'])[mask].square().mean();pair=mask[1:]&mask[:-1]
            if pair.any():loss=loss+.1*((pred.diff(dim=0)-b['target'].diff(dim=0))[pair]).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),2.)
            if not torch.isfinite(grad):raise FloatingPointError('nonfinite query-local gradient')
            opt.step()
            with torch.no_grad():
                emap=dict(ema.named_parameters())
                for name,param in model.named_parameters():
                    if param.requires_grad:emap[name].mul_(.999).add_(param,alpha=.001)
            if step==1 or step%100==0 or step==args.steps:
                row=dict(step=step,seconds=elapsed+time.monotonic()-start,loss=float(loss.detach()),
                    batch_rmse_mm=float((pred-b['target'])[mask].square().mean().sqrt().detach()*data.std[0]))
                trace.append(row);write_json(folder/'progress.json',row);write_json(folder/'curve.json',trace)
                print(json.dumps(dict(seed=seed,**row)),flush=True)
            if step%500==0:
                torch.save(dict(seed=seed,step=step,lr=args.lr,model=model.state_dict(),ema=ema.state_dict(),
                    optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),trace=trace,data_sha256=sha(args.data/'contract.json'),
                    source_sha256=sha(args.source/f'gru_s{seed}/last.pt'),code_sha256=protocol['code_sha256']),folder/f'checkpoint_{step}.pt')
                torch.cuda.empty_cache()
        for label,net in [('Qraw',model),('Qema',ema)]:
            dest=out/f'{label}_s{seed}';dest.mkdir(exist_ok=False);torch.save(net.state_dict(),dest/'last.pt')
            write_json(dest/'fit.json',dict(seed=seed,case=label,steps=args.steps,trace=trace,parameters=sum(p.numel() for p in net.parameters()),same_training_trajectory=True))
            metrics=evaluate(data,net,72,12,dest);row=dict(seed=seed,case=label,steps=args.steps,metrics=metrics)
            summary.append(row);write_json(out/'summary.json',summary);print(json.dumps(row),flush=True)
        del model,ema,opt;torch.cuda.empty_cache()
    write_json(out/'status.json',dict(status='COMPLETE_QUERY_LOCAL_2_SEEDS',completed=len(summary)))


if __name__=='__main__':main()
