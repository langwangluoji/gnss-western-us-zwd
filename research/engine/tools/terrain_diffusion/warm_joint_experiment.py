"""Preserve the ~10mm reference exactly, then train new fusion in matched cohorts."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .data import PROJECT,DEFAULT_DATA,sha,write_json
from .joint_data import JointData
from .p14_fusion import P14Fusion,chronological_blocks
from .masked_transformer_fusion import MaskedP14Fusion
from .p14_fusion_experiment import evaluate


class WarmData(JointData):
    def __init__(self,path):
        super().__init__(path)
        with np.load(DEFAULT_DATA/'data.npz') as old:
            self.mean=old['mean'];self.std=old['std']
        self.mean_t=torch.as_tensor(self.mean,device='cuda');self.std_t=torch.as_tensor(self.std,device='cuda')
        for f in (self.grid_features,self.train_features,self.dev_features,self.small_features):f.mean=self.mean_t;f.std=self.std_t
        self.legacy_grid=torch.as_tensor(self.events.grid,device='cuda',dtype=torch.float32)
        # This is the refit/development lane, not inner2019 or strict OOF.
        # Reproduce the baseline's full causal observed-age axis exactly.
        self.set_split(2016,2021)

    @torch.no_grad()
    def batch(self,anchors,context,development=False,temporal=True,features=True):
        b=super().batch(anchors,context,development,True,True)
        start,end=int(anchors[0]),int(anchors[-1]);c=torch.as_tensor(context,device='cuda')
        raw=self.values[start-self.length+1:end+1,c]
        namespace=SimpleNamespace(length=self.length,small_grid=self.legacy_grid,train_features=self.train_features,side=8)
        b['legacy_history']=JointData.temporal_history(namespace,raw,self.hours[start-self.length+1:end+1],c,len(anchors),
            self.station_age[start-self.length+1:end+1,c])
        return b


class Reference(nn.Module):
    def __init__(self,state,kind,length):
        super().__init__();self.inner=P14Fusion(kind,length);self.inner.load_state_dict(state)

    @property
    def base(self):return self.inner.base

    def forward(self,b):return self.inner(dict(b,history=b['legacy_history']))


class WarmJoint(MaskedP14Fusion):
    def __init__(self,state,source_kind='gru',kind='T5',length=72,side=16,width=64,joint=False):
        super().__init__(kind,length,side,width)
        self.reference=Reference(state,source_kind,length)
        for p in self.reference.parameters():p.requires_grad_(joint)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=PROJECT/'results/20261125_temporal_continuous_data')
    p.add_argument('--source',type=Path,default=PROJECT/'results/20261125_p14_continuous_fusion')
    p.add_argument('--source-kind',default='gru');p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,17]);p.add_argument('--cases',nargs='+',default=['T2_frozen','T5_frozen','T5_joint'])
    p.add_argument('--eval-hours',type=int,default=72);p.add_argument('--months',type=int,default=12)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    protocol=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        code_sha256={f:sha(Path(__file__).with_name(f)) for f in ('warm_joint_experiment.py','joint_data.py','masked_transformer_fusion.py','p14_fusion.py','p14_fusion_experiment.py')},
        data_sha256=sha(args.data/'contract.json'),
        objective='Improve matched ~10mm best p14 fusion; preserve exact initialized reference',
        source_weights={str(s):sha(args.source/f'{args.source_kind}_s{s}/last.pt') for s in args.seeds},
        training='2016-2019 continuous chronological12h blocks; only station IDs sampled; all target hours covered once before epoch restart',
        task='R current field reconstruction; frozen full-history reference, not strict OOF or independent2019 selection',
        evaluation='2020 monthlydays8-10 x3layouts x137dev; already development-exposed; final3000step primary',
        learning_rates=dict(new=1e-5,reference_joint=1e-6),confirmation_C_D_read=False)
    path=args.out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise ValueError('Existing run lineage differs')
    else:write_json(path,protocol)
    data=WarmData(args.data);e=data.events
    stop=int(np.searchsorted(e.hours,np.datetime64('2020-01-01','h').astype('int64')))
    blocks=chronological_blocks(71,stop,12)
    summary=json.loads((args.out/'summary.json').read_text()) if (args.out/'summary.json').exists() else []
    for seed in args.seeds:
        state=torch.load(args.source/f'{args.source_kind}_s{seed}/last.pt',weights_only=True)
        for case in ['reference']+args.cases:
            if any(r['seed']==seed and r['case']==case for r in summary):continue
            folder=args.out/f'{case}_s{seed}';folder.mkdir(exist_ok=False)
            torch.manual_seed(seed);kind=case.split('_')[0] if case!='reference' else 'T5'
            model=WarmJoint(state,args.source_kind,kind,joint=case.endswith('joint')).cuda()
            rng=np.random.default_rng(seed+28011);trace=[];start=time.monotonic()
            # Every initialized candidate equals the exact old GRU foundation.
            probe=data.batch(np.arange(71,73),e.layouts[0])
            with torch.no_grad():
                initial=model(probe);ref=model.reference(probe)
                if not torch.equal(initial,ref):raise ValueError('warm start does not preserve old reference')
            if case!='reference':
                old=list(model.reference.parameters());old_ids={id(p) for p in old}
                new=[p for p in model.parameters() if id(p) not in old_ids and p.requires_grad]
                groups=[dict(params=new,lr=1e-5)]
                if case.endswith('joint'):groups.append(dict(params=old,lr=1e-6))
                opt=torch.optim.AdamW(groups,weight_decay=1e-4)
                for step in range(1,args.steps+1):
                    anchors=blocks[(step-1)%len(blocks)];c=np.sort(rng.choice(410,82,False))
                    b=data.batch(anchors,c);model.train();prediction=model(b);mask=b['mask']
                    loss=(prediction-b['target'])[mask].square().mean();pair=mask[1:]&mask[:-1]
                    if pair.any():loss=loss+.1*((prediction.diff(dim=0)-b['target'].diff(dim=0))[pair]).square().mean()
                    opt.zero_grad(set_to_none=True);loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),2.)
                    if not torch.isfinite(grad):raise FloatingPointError('nonfinite warm fusion gradient')
                    opt.step()
                    if step==1 or step%100==0 or step==args.steps:
                        row=dict(step=step,first_hour=int(e.hours[anchors[0]]),last_hour=int(e.hours[anchors[-1]]),
                            batch_rmse_mm=float((prediction-b['target'])[mask].square().mean().sqrt().detach()*data.std[0]),
                            loss=float(loss.detach()),seconds=time.monotonic()-start)
                        trace.append(row);write_json(folder/'progress.json',row);print(json.dumps(dict(case=case,seed=seed,**row)),flush=True)
                    if step%1000==0:torch.save(model.state_dict(),folder/f'step_{step}.pt')
                torch.save(model.state_dict(),folder/'last.pt')
                write_json(folder/'fit.json',dict(seed=seed,case=case,steps=args.steps,trace=trace,
                    seconds=time.monotonic()-start,initialization_exact=True,parameters=sum(p.numel() for p in model.parameters())))
                del opt
            metrics=evaluate(data,model,args.eval_hours,args.months,folder)
            row=dict(seed=seed,case=case,steps=0 if case=='reference' else args.steps,metrics=metrics)
            summary.append(row);write_json(args.out/'summary.json',summary)
            print(json.dumps(dict(stage='evaluated',**row)),flush=True);del model;torch.cuda.empty_cache()
    write_json(args.out/'status.json',dict(status='COMPLETE_MATCHED_WARM_FUSION',results=len(summary)))


if __name__=='__main__':main()
