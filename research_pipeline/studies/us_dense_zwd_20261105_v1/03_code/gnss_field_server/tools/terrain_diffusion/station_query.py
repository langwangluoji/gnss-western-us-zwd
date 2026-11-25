"""Native-station causal encoders and continuous local query attention."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import PROJECT,sha,write_json
from .p14_fusion import FusionData,chronological_blocks
from .warm_joint_experiment import Reference
from .p14_fusion_experiment import evaluate


def hole_schedule(steps,seed):
    rng=np.random.default_rng(seed+55119);result=np.zeros(steps,dtype=bool)
    for start in range(0,steps,4):result[start+int(rng.integers(min(4,steps-start)))]=True
    return result


def station_tokens(raw,hours,ages,mean,std,length=72):
    if len(raw)<length:raise ValueError('insufficient history')
    if not torch.all(torch.diff(hours)==1):raise ValueError('history must be hourly')
    mask=torch.isfinite(raw);norm=torch.where(mask,(raw-mean)/std,0)
    h=norm.unfold(0,length,1).permute(0,3,1,2)
    m=mask.unfold(0,length,1).permute(0,3,1,2).float()
    age=ages.unfold(0,length,1).permute(0,2,1).log1p()[...,None]/5
    hh=hours.unfold(0,length,1).double()
    calendar=torch.stack([(hh%24*2*torch.pi/24).sin(),(hh%24*2*torch.pi/24).cos(),
        (hh*2*torch.pi/(365.25*24)).sin(),(hh*2*torch.pi/(365.25*24)).cos()],-1).float()
    history=torch.cat([h,m,age,calendar[:,:,None].expand(-1,-1,raw.shape[1],-1)],-1)
    z=raw[:,:,0].unfold(0,length,1) # B,C,L
    index=torch.arange(length,device=raw.device)[None,None]
    last=torch.where(torch.isfinite(z),index,-1).max(-1).values
    latest=z.gather(-1,last.clamp_min(0)[...,None]).squeeze(-1)
    return dict(history=history,latest=torch.nan_to_num(latest),available=last>=0)


class StationData(FusionData):
    @torch.no_grad()
    def batch(self,anchors,context,development=False,temporal=True,features=True):
        b=super().batch(anchors,context,development,True,True);b['legacy_history']=b['history']
        a,z=int(anchors[0]),int(anchors[-1]);c=torch.as_tensor(context,device='cuda')
        raw=self.values[a-self.length+1:z+1,c]
        t=station_tokens(raw,self.hours[a-self.length+1:z+1],self.station_age[a-self.length+1:z+1,c],self.mean_t,self.std_t,self.length)
        b['station_history']=t.pop('history');b.update(t)
        b['condition_static']=torch.cat([self.train_features.static[c],self.coast[c,None].log1p()/6],-1)
        g=self.events.geo;device=self.values.device
        prefix='dev_' if development else ''
        b['qxy']=torch.as_tensor(g[prefix+'xy'],device=device,dtype=torch.float32)
        b['qheight']=torch.as_tensor(g[prefix+'height'],device=device,dtype=torch.float32)
        b['qcoast']=torch.as_tensor(g[prefix+'coast'],device=device,dtype=torch.float32)
        b['cxy']=torch.as_tensor(g['xy'][context],device=device,dtype=torch.float32)
        b['cheight']=torch.as_tensor(g['height'][context],device=device,dtype=torch.float32)
        b['ccoast']=self.coast[c];b['age']=self.station_age[torch.as_tensor(anchors,device=device)[:,None],c[None]]
        b['mean']=self.mean_t;b['std']=self.std_t
        return b


class StationTemporal(nn.Module):
    def __init__(self,kind='transformer',width=32):
        super().__init__();self.kind=kind;self.width=width;self.static=nn.Linear(8,width)
        if kind in ('gru','lstm'):
            self.time=(nn.GRU if kind=='gru' else nn.LSTM)(11,width,batch_first=True)
        elif kind=='transformer':
            self.project=nn.Linear(11,width)
            layer=nn.TransformerEncoderLayer(width,4,width*2,dropout=0,batch_first=True,norm_first=True)
            self.time=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        else:raise ValueError(kind)

    def forward(self,x,static,all_times=False):
        B,T,C,_=x.shape;sequence=x.permute(0,2,1,3).reshape(B*C,T,11)
        s=self.static(static)[None].expand(B,-1,-1).reshape(B*C,self.width)
        if self.kind in ('gru','lstm'):
            h=s.tanh()[None];state=(h,torch.zeros_like(h)) if self.kind=='lstm' else h
            out,_=self.time(sequence,state)
        else:
            position=torch.arange(T,device=x.device,dtype=x.dtype)[:,None]
            frequency=torch.exp(torch.arange(0,self.width,2,device=x.device)*(-math.log(10000)/self.width))
            pe=torch.zeros(T,self.width,device=x.device);pe[:,0::2]=(position*frequency).sin();pe[:,1::2]=(position*frequency).cos()
            mask=torch.ones(T,T,device=x.device,dtype=torch.bool).triu(1)
            out=self.time(self.project(sequence)+s[:,None]+pe[None],mask=mask)
        out=out.reshape(B,C,T,self.width).permute(0,2,1,3)
        return out if all_times else out[:,-1]


class StationResidual(nn.Module):
    def __init__(self,state,kind='transformer',length=72):
        super().__init__();self.kind='S';self.reference=Reference(state,'gru',length)
        for p in self.reference.parameters():p.requires_grad_(False)
        self.time=StationTemporal(kind)
        self.state_norm=nn.LayerNorm(32);self.query_norm=nn.LayerNorm(32);self.key_norm=nn.LayerNorm(32)
        self.high_norm=nn.LayerNorm(16);self.low_norm=nn.LayerNorm(32)
        self.query=nn.Linear(171,32);self.key=nn.Linear(42,32)
        self.value=nn.Sequential(nn.Linear(42,64),nn.SiLU(),nn.Linear(64,32))
        self.local=nn.Sequential(nn.Linear(204,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)

    def forward(self,b):
        reference=self.reference(b);B,N=reference.shape
        state=self.state_norm(self.time(b['station_history'],b['condition_static']))
        distance=torch.cdist(b['qxy'].float(),b['cxy'].float(),compute_mode='donot_use_mm_for_euclid_dist')[None].expand(B,-1,-1)
        distance=distance.masked_fill(~b['available'][:,None],float('inf'))
        d,index=distance.topk(min(16,state.shape[1]),largest=False);valid=torch.isfinite(d)
        batch=torch.arange(B,device=state.device)[:,None,None]
        selected=state[batch,index];d=d.clamp_max(1000)
        offset=(b['cxy'][index]-b['qxy'][None,:,None])/100
        dh=b['cheight'][index]-b['qheight'][None,:,None]
        coast=(b['ccoast'][index]-b['qcoast'][None,:,None])/500
        surface=reference*b['std'][0]+b['mean'][0]
        innovation=((b['latest'][batch,index]*torch.exp(dh/3000)-surface[...,None])/b['std'][0]).float()
        age=b['age'][batch,index].log1p()/5
        terrain=b['condition_static'][index][...,3:6]-b['static'][:,:,None,3:6]
        edge=torch.cat([offset,d.log1p()[...,None]/6,dh[...,None]/1000,coast[...,None],
            innovation[...,None],age[...,None],terrain],-1).float()
        message=torch.cat([selected,edge],-1)
        # Reuse both frozen U-Net scales at the physical query coordinates.
        # Upsampling uses the same pixel-center rule as the original carrier.
        carrier=self.reference.base;f0=carrier.proj(b['support']);f1=carrier.down(F.avg_pool2d(f0,2))
        f1=F.interpolate(f1,size=f0.shape[-2:],mode='bilinear',align_corners=False)
        sample=lambda field:F.grid_sample(field,b['uv'],padding_mode='border',align_corners=True)[:,:,0].transpose(1,2)
        multiscale=torch.cat([self.high_norm(sample(f0)),self.low_norm(sample(f1))],-1)
        qfeature=torch.cat([b['point'],b['static'],reference[...,None],multiscale],-1)
        logits=(self.query_norm(self.query(qfeature))[:,:,None]*self.key_norm(self.key(message))).sum(-1)/math.sqrt(32)-torch.log1p(d/50)
        weight=logits.masked_fill(~valid,-1e4).softmax(-1)*valid
        weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
        aggregate=(self.value(message)*weight[...,None]).sum(-2)
        physical=(innovation*weight).sum(-1,keepdim=True)
        raw=self.local(torch.cat([qfeature,aggregate,physical],-1)).squeeze(-1)
        limit=.05*b.get('time_scale',1.)
        return reference+limit*torch.tanh(raw/limit)*valid.any(-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,default=PROJECT/'results/20261125_temporal_continuous_data')
    p.add_argument('--source',type=Path,default=PROJECT/'results/20261125_p14_continuous_fusion')
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--seeds',type=int,nargs='+',default=[42,17])
    p.add_argument('--kind',choices=['gru','lstm','transformer'],default='transformer');args=p.parse_args()
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.32)
    args.out.mkdir(parents=True,exist_ok=False);out=args.out
    write_json(out/'protocol.json',dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_weights={str(s):sha(args.source/f'gru_s{s}/last.pt') for s in args.seeds},
        data_sha256=sha(args.data/'contract.json'),code_sha256={f:sha(Path(__file__).with_name(f)) for f in
            ('station_query.py','warm_joint_experiment.py','p14_fusion.py','temporal_encoders.py','fixed_support.py','unet_foundation.py','data.py')},
        source_kind='gru',training='continuous2016-2019,12h blocks,82station IDs only sampled;75%random/25%hole100',
        new_lr=1e-4,ema=.999,trust_region_mm=5,raw_and_ema_final_preregistered=True,
        evaluation='same2020monthly72h x3layouts x137dev, exposed',strict_oof=False,confirmation_C_D_read=False))
    data=StationData(args.data);e=data.events;stop=int(np.searchsorted(e.hours,np.datetime64('2020-01-01','h').astype('int64')))
    blocks=chronological_blocks(71,stop,12);summary=[];xy=e.geo['xy']
    for seed in args.seeds:
        torch.manual_seed(seed);rng=np.random.default_rng(seed+28011)
        holes=hole_schedule(args.steps,seed)
        source=torch.load(args.source/f'gru_s{seed}/last.pt',weights_only=True)
        model=StationResidual(source,args.kind).cuda();ema=copy.deepcopy(model).eval()
        for module in ema.modules():
            if isinstance(module,nn.RNNBase):module.flatten_parameters()
        folder=out/f'trajectory_s{seed}';folder.mkdir();trace=[];start=time.monotonic()
        probe=data.batch(np.arange(71,73),e.layouts[0])
        with torch.no_grad():
            pred=model(probe);ref=model.reference(probe)
            if not torch.equal(pred,ref):raise ValueError('native station branch changed initialized reference')
            if not torch.isfinite(probe['station_history']).all():raise ValueError('nonfinite station history')
        write_json(folder/'initial_probe.json',dict(exact_reference=True,history_shape=list(probe['station_history'].shape),input_finite=True))
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
        for step in range(1,args.steps+1):
            anchors=blocks[(step-1)%len(blocks)]
            if holes[step-1]:
                center=int(rng.integers(410));pool=np.flatnonzero(np.linalg.norm(xy-xy[center],axis=1)>100)
            else:pool=np.arange(410)
            c=np.sort(rng.choice(pool,82,False));b=data.batch(anchors,c);pred=model(b);mask=b['mask']
            loss=(pred-b['target'])[mask].square().mean();pair=mask[1:]&mask[:-1]
            if pair.any():loss=loss+.1*((pred.diff(dim=0)-b['target'].diff(dim=0))[pair]).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),2.)
            if not torch.isfinite(grad):raise FloatingPointError('nonfinite station-query gradient')
            opt.step()
            with torch.no_grad():
                emap=dict(ema.named_parameters())
                for name,param in model.named_parameters():
                    if param.requires_grad:emap[name].mul_(.999).add_(param,alpha=.001)
            if step==1 or step%100==0 or step==args.steps:
                row=dict(step=step,seconds=time.monotonic()-start,loss=float(loss.detach()),
                    batch_rmse_mm=float((pred-b['target'])[mask].square().mean().sqrt().detach()*data.std[0]))
                trace.append(row);write_json(folder/'curve.json',trace);write_json(folder/'progress.json',row)
                print(json.dumps(dict(seed=seed,**row)),flush=True)
        for label,net in [(f'S_{args.kind}_raw',model),(f'S_{args.kind}_ema',ema)]:
            dest=out/f'{label}_s{seed}';dest.mkdir();torch.save(net.state_dict(),dest/'last.pt')
            write_json(dest/'fit.json',dict(case=label,seed=seed,steps=args.steps,trace=trace,same_training_trajectory=True))
            metrics=evaluate(data,net,72,12,dest);row=dict(seed=seed,case=label,steps=args.steps,metrics=metrics)
            summary.append(row);write_json(out/'summary.json',summary);print(json.dumps(row),flush=True)
        del model,ema,opt;torch.cuda.empty_cache()
    write_json(out/'status.json',dict(status='COMPLETE_NATIVE_STATION_QUERY',completed=len(summary)))


if __name__=='__main__':main()
