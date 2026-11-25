"""Executable T/L/A/B/J pipeline. Arrays/weights local, receipts and plots public.

Example (engineering pilot, explicitly not the 3000-step formal experiment):
python -m tools.terrain_diffusion.joint_experiment --out results/NAME --diagnostic \
    --steps 60 --oof-steps 60 --representation-steps 100 --probability-steps 100 \
    --train-hours 120 --eval-hours 24 --members 16
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.fft import dct
from .data import PROJECT,sha,write_json
from .joint_data import JointData,fold_groups
from .p14_fusion import chronological_blocks
from .masked_transformer_fusion import MaskedP14Fusion
from .joint_probability import ProbabilityCondition,InnovationDiffusion,ConditionalGaussian,LatentField
from .spectral_operators import AnchorField,InnovationTransform,history_spectrum,joint_transform,require_forecast_origin


def enforce_representation_gate(error,diagnostic=False):
    if not np.isfinite(error) or (error>1 and not diagnostic):
        raise RuntimeError(f'representation reconstruction {error:.3f}mm exceeds 1mm gate')


def check_family_gate(metrics,diagnostic):
    reconstruction=max(metrics['reconstruction_train_mm'],metrics['reconstruction_inner2019_mm'])
    spatial_ok=np.isfinite(metrics['masked_label_reconstruction_mm']) and metrics['masked_events']>0
    passed=bool(np.isfinite(reconstruction) and reconstruction<=1 and spatial_ok)
    if not passed and not diagnostic:raise RuntimeError('representation family failed 1mm train/inner reconstruction or spatial-check eligibility')
    return dict(passed=passed,diagnostic_override=bool(not passed and diagnostic),
        threshold_mm=1.,metrics=metrics,spatial_rule='finite held-out-label check with nonempty cohort; error reported separately')


def verify_lineage(root,diagnostic,data_hash,code_hashes):
    for path in Path(root).glob('protocol_*.json'):
        prior=json.loads(path.read_text())
        if prior['config']['diagnostic']!=diagnostic:raise ValueError('upstream diagnostic/formal mode differs; use a new output')
        if prior['data_sha256']!=data_hash:raise ValueError('upstream data differs')
        for name,digest in prior['code_sha256'].items():
            if code_hashes.get(name)!=digest:raise ValueError(f'upstream code differs: {name}; use a new output')


def forecast_targets(values,origin,mean_at_origin):
    require_forecast_origin(np.array([origin]),np.array([origin]),np.array([origin]))
    return values[origin+1:origin+7],np.repeat(mean_at_origin[None],6,axis=0)


def past_spectral_condition(history):
    values=np.where(history[:,:,3]>0,history[:,:,0],np.nan)
    spec=history_spectrum(values,len(values)-1).reshape(values.shape[1],4,4)
    weight=spec[:,:,3];den=weight.sum(0).clip(1e-8)
    out=np.zeros((4,4),dtype='float32')
    out[:,:3]=(spec[:,:,:3]*weight[:,:,None]).sum(0)/den[:,None]
    out[:,3]=weight.mean(0)
    return out.ravel()


def fair_crps(samples,truth):
    K=len(samples)
    if K<2:raise ValueError('fair CRPS requires two or more members')
    x=np.sort(samples,axis=0)
    weights=(2*np.arange(K)-K+1).reshape(K,*([1]*(x.ndim-1)))
    return np.mean(np.abs(samples-truth),axis=0)-(x*weights).sum(0)/(K*(K-1))


def scores(samples,truth):
    if not np.isfinite(samples).all():raise ValueError('nonfinite samples')
    error=samples.mean(0)-truth;ok=np.isfinite(truth)
    def macro(a):
        count=np.isfinite(a).sum((0,1));valid=count>0
        return float(np.sqrt(np.nansum(a*a,axis=(0,1))[valid]/count[valid]).mean())
    inc=np.diff(samples,axis=2);inc_truth=np.diff(truth,axis=1)
    lo,hi=np.quantile(samples,[.05,.95],axis=0)
    crps=fair_crps(samples,truth)
    # Station macro, with layouts/events equally pooled in this short cohort.
    count=np.isfinite(crps).sum((0,1));valid=count>0
    return dict(macro_rmse_mm=macro(error),micro_rmse_mm=float(np.sqrt(np.nanmean(error**2))),
        mae_mm=float(np.nanmean(np.abs(error))),bias_mm=float(np.nanmean(error)),
        fair_crps_station_macro_mm=float((np.nansum(crps,(0,1))[valid]/count[valid]).mean()),
        increment_macro_rmse_mm=macro(inc.mean(0)-inc_truth),
        increment_fair_crps_mm=float(np.nanmean(fair_crps(inc,inc_truth))),
        coverage90=float(np.mean(((truth>=lo)&(truth<=hi))[ok])),
        width90_mm=float((hi-lo)[ok].mean()),valid_count=int(ok.sum()))


def train_foundation(data,kind,steps,seed,folder,batch_hours=6):
    folder.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed)
    model=MaskedP14Fusion(kind,side=data.side).cuda()
    # Fresh model is required here: old p14 weights saw 2019/outer stations.
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    blocks=chronological_blocks(data.split_start+max(data.length,12)-1,data.split_stop,batch_hours)
    rng=np.random.default_rng(seed+28011);trace=[];start=time.monotonic()
    for step in range(1,steps+1):
        anchors=blocks[(step-1)%len(blocks)]
        c=np.sort(rng.choice(data.train_ids,min(82,round(.2*len(data.train_ids))),False))
        b=data.batch(anchors,c);p=model(b);mask=b['mask']
        if not mask.any():raise ValueError('training block has no target labels')
        loss=(p-b['target'])[mask].square().mean()
        pair=mask[1:]&mask[:-1]
        if pair.any():loss=loss+.1*((p.diff(dim=0)-b['target'].diff(dim=0))[pair]).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),2.)
        if not torch.isfinite(grad):raise FloatingPointError('foundation gradient nonfinite')
        opt.step()
        if step==1 or step%20==0 or step==steps:
            row=dict(step=step,first_hour=int(data.events.hours[anchors[0]]),last_hour=int(data.events.hours[anchors[-1]]),
                batch_rmse_mm=float((p-b['target'])[mask].square().mean().sqrt().detach()*data.std[0]),
                loss=float(loss.detach()),seconds=time.monotonic()-start)
            trace.append(row);write_json(folder/'progress.json',row);print(json.dumps(dict(kind=kind,**row)),flush=True)
    torch.save(model.state_dict(),folder/'last.pt')
    write_json(folder/'fit.json',dict(kind=kind,steps=steps,seed=seed,trace=trace,
        parameters=sum(p.numel() for p in model.parameters()),weights_sha256=sha(folder/'last.pt'),
        mean=data.mean.tolist(),std=data.std.tolist(),train_ids=data.train_ids.tolist(),
        split=[data.split_start,data.split_stop],fresh_start=True,chronological=True))
    return model.eval()


@torch.no_grad()
def foundation_monitor(data,model,hours=24):
    data.set_split(2019,2020);errors=[];allmask=[]
    start=data.split_start+7*24
    for a in range(start,start+hours,6):
        c=data.train_ids[np.random.default_rng(8120).permutation(len(data.train_ids))[:min(82,len(data.train_ids)//5)]]
        b=data.batch(np.arange(a,min(a+6,start+hours)),np.sort(c))
        errors.append(((model(b)-b['target'])*data.std[0]).cpu().numpy());allmask.append(b['mask'].cpu().numpy())
    err=np.where(np.concatenate(allmask),np.concatenate(errors),np.nan)
    valid=np.isfinite(err).sum(0)>0
    macro=np.sqrt(np.nansum(err**2,axis=0)[valid]/np.isfinite(err).sum(0)[valid]).mean()
    return dict(macro_rmse_mm=float(macro),micro_rmse_mm=float(np.sqrt(np.nanmean(err**2))),
        year=2019,first_day='January08',hours=hours,role='training-pool hidden groups; inner diagnostic')


@torch.no_grad()
def collect_fold(data,model,outer,start,hours,folder,tag,forecast=False):
    """Keep every (outer-fold,layout,6h start) as one event; never merge folds."""
    e=data.events;keep=data.train_ids
    c=np.sort(keep[np.random.default_rng(8120).permutation(len(keep))[:min(82,len(keep)//5)]])
    payload={k:[] for k in ('history','mu','truth','residual','spectrum','clock','hours','mu_grid','origin_hours','source_hours','mean_source_hours')}
    grid_means=[];history=[];means=[];truth=[]
    # Each mu frame is computed with its own exact 72-hour observed window.
    for a in range(start-data.length+1,start+hours,6):
        anchors=np.arange(a,min(a+6,start+hours));b=data.batch(anchors,c)
        mean=(model(b)*data.std[0]+data.mean[0]).cpu().numpy()
        grid=(model(data.query_batch(anchors,c,b))*data.std[0]+data.mean[0]).cpu().numpy()
        grid_means.extend(grid);history.extend(b['history'][:,-1].cpu().numpy())
        means.extend(mean);truth.extend(np.asarray(e.values[anchors,:,0]))
    gm=np.asarray(grid_means);hist=np.asarray(history);means=np.asarray(means);truth=np.asarray(truth)
    grid26=np.concatenate([hist,gm[:,:,None]/100],-1)
    for offset in range(0,hours-5,6):
        i=data.length-1+offset;physical_start=start+offset
        if forecast:
            # Forecast origin is the last observed frame. Persistence is a
            # transparent origin-only mean control, not a trained F mean.
            if offset+6>=hours:break
            target,futuremean=forecast_targets(e.values[:,:,0],physical_start,means[i])
            mean=futuremean;actual=np.array(target)
            hs=np.repeat(grid26[i-data.length+1:i+1][None],6,axis=0)
        else:
            mean=means[i:i+6];actual=truth[i:i+6]
            hs=np.stack([grid26[j-data.length+1:j+1] for j in range(i,i+6)])
        masked=actual.copy();masked[:,np.setdiff1d(np.arange(410),outer)]=np.nan
        payload['history'].append(hs.astype('float16'))
        payload['mu'].append(mean.astype('float32'));payload['truth'].append(masked.astype('float32'))
        payload['residual'].append((masked-mean).astype('float32'))
        spectra=[]
        for j in range(6):
            end=i if forecast else i+j
            spectra.append(past_spectral_condition(hist[end-data.length+1:end+1]))
        payload['spectrum'].append(np.asarray(spectra))
        label_start=physical_start+1 if forecast else physical_start
        hh=np.asarray(e.hours[label_start:label_start+6],float)
        clock=np.c_[np.arange(6)/5,(hh%24)/24,np.array([1,0,0,0,0,0])]
        payload['clock'].append(clock.astype('float32'));payload['hours'].append(hh.astype('int64'))
        origins=np.repeat(e.hours[physical_start],6) if forecast else hh.astype('int64')
        source=np.stack([e.hours[physical_start-data.length+1+(0 if forecast else j):physical_start+1+(0 if forecast else j)] for j in range(6)])
        require_forecast_origin(origins,source.max(-1),origins)
        payload['origin_hours'].append(origins);payload['source_hours'].append(source);payload['mean_source_hours'].append(origins)
        payload['mu_grid'].append(gm[i:i+6].astype('float32') if not forecast else np.repeat(gm[i:i+1],6,axis=0))
    arrays={k:np.asarray(v) for k,v in payload.items()}
    arrays.update(static=data.time_static.cpu().numpy(),query_static=np.c_[e.geo['static'],np.log1p(e.geo['coast'])/6,e.geo['coast']/500],
        query_xy=e.geo['xy'],support_xy=data.small_grid.cpu().numpy(),outer=outer,condition_ids=c)
    path=folder/f'{tag}.npz';np.savez_compressed(path,**arrays)
    write_json(folder/f'{tag}_receipt.json',dict(file=path.name,sha256=sha(path),events=len(arrays['hours']),
        outer_ids=outer.tolist(),condition_ids=c.tolist(),event_key='fold/layout0/six_hour_start',
        history_length=data.length,forecast=forecast,last_source='origin' if forecast else 'current frame',
        task='F origin-persistence mean diagnostic' if forecast else 'R current-hour reconstruction'))


def load_events(root,tag):
    folders=sorted(root.glob('fold*'))
    if [f.name for f in folders]!=[f'fold{i}' for i in range(5)]:raise ValueError('exactly five OOF folds required')
    packs=[]
    for folder in folders:
        path=folder/f'{tag}.npz';receipt=json.loads((folder/f'{tag}_receipt.json').read_text())
        if sha(path)!=receipt['sha256']:raise ValueError('OOF cache hash mismatch')
        pack=dict(np.load(path))
        if set(pack['outer'])&set(pack['condition_ids']):raise ValueError('OOF condition/outer overlap')
        require_forecast_origin(pack['origin_hours'],pack['source_hours'].max(-1),pack['mean_source_hours'])
        if not np.all(np.diff(pack['source_hours'],axis=-1)==1):raise ValueError('OOF history not hourly')
        packs.append(pack)
    if sorted(np.concatenate([p['outer'] for p in packs]).tolist())!=list(range(410)):raise ValueError('OOF folds do not partition410')
    for p in packs[1:]:
        for k in ('static','query_static','query_xy','support_xy'):
            if not np.array_equal(p[k],packs[0][k]):raise ValueError(f'OOF geometry mismatch: {k}')
    keys=('history','mu','truth','residual','spectrum','clock','hours','mu_grid','origin_hours','source_hours','mean_source_hours')
    result={k:np.concatenate([p[k] for p in packs]) for k in keys}
    # Concatenating EVENT axis is valid; label sets are never combined in a field.
    result.update({k:packs[0][k] for k in ('static','query_static','query_xy','support_xy')})
    # Recompute from stored true coverage; cached zero-filled means are never
    # interpreted as observed zeros by the harmonic fit.
    result['spectrum']=np.stack([[past_spectral_condition(h) for h in event] for event in result['history']])
    result['fold']=np.concatenate([np.full(len(p['hours']),i) for i,p in enumerate(packs)])
    return result


def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')


def representation(root,train,valid,args):
    folder=root/'representation';folder.mkdir(exist_ok=False)
    bias=float(np.nanmean(train['residual']));scale=max(float(np.nanstd(train['residual'])),.1)
    x=np.linspace(train['support_xy'][:,0].min(),train['support_xy'][:,0].max(),16)
    y=np.linspace(train['support_xy'][:,1].min(),train['support_xy'][:,1].max(),16)
    xx,yy=np.meshgrid(x,y);field=AnchorField(np.c_[xx.ravel(),yy.ravel()])
    H,outside=field.readout(train['query_xy'],True)
    u=(train['residual']-bias)/scale;uv=(valid['residual']-bias)/scale
    a=field.encode(u,H);av=field.encode(uv,H)
    error=(a@H.T-u)*scale;ev=(av@H.T-uv)*scale
    # Same event, but encode only half its available physical labels for a
    # distinct interpolation check (not a claim of dense ground truth).
    masked=u.copy();masked[:,:,::2]=np.nan
    # Some folds may have too few labels at a missing epoch. Keep only events
    # with at least one encoding label for all six frames, report denominator.
    good=np.isfinite(masked).any(-1).all(-1)
    cross=field.encode(masked[good],H)@H.T
    cross_error=(cross-u[good])*scale;cross_error[:,:,1::2]=np.nan
    anchor_metrics=dict(reconstruction_train_mm=float(np.sqrt(np.nanmean(error**2))),
        reconstruction_inner2019_mm=float(np.sqrt(np.nanmean(ev**2))),
        masked_label_reconstruction_mm=float(np.sqrt(np.nanmean(cross_error**2))),
        masked_events=int(good.sum()),outside_hull_stations=int(outside.sum()),
        note='Available OOF labels only, no dense truth; ridge=.01, smooth=.01 fixed before evaluation')
    np.savez_compressed(folder/'anchor.npz',xy=field.xy,U=field.U,L=field.L,H=H,eigenvalues=field.eigenvalues,
        train=a,inner=av,bias=bias,scale=scale)
    write_json(folder/'anchor_metrics.json',anchor_metrics)
    # L-family trained label encoder/basis. G is a separate shared encoder,
    # never copied from the fold-specific foundation.
    torch.manual_seed(args.seed);condition=ProbabilityCondition().cuda()
    st=tensor(train['query_static']);ts=tensor(train['static']);trace=[]
    def get(pack,i,training):
        h=tensor(pack['history'][i]);m=tensor(pack['mu'][i])/100
        hm=condition.encoder(h,ts)
        xy=tensor((pack['query_xy']-pack['support_xy'].min(0))/(pack['support_xy'].max(0)-pack['support_xy'].min(0))*2-1)
        q=F.grid_sample(hm,xy[None,None].expand(6,-1,-1,-1),align_corners=True,padding_mode='border')[:,:,0].transpose(1,2)
        features=torch.cat([st[None].expand(6,-1,-1)[:,:,:8],q,m[...,None]],-1)
        labels=tensor((pack['residual'][i]-bias)/scale);mask=torch.isfinite(labels)
        z=latent.encode(labels,st,mask);pred=latent.decode(z,features)
        return z,pred,labels,mask
    # Phi has 8 static +64 shared G +mu =73 dimensions.
    latent=LatentField(condition_width=63).cuda()
    opt=torch.optim.AdamW(list(latent.parameters())+list(condition.parameters()),lr=3e-4)
    for step in range(args.representation_steps):
        i=step%len(u);z,pred,labels,mask=get(train,i,True)
        loss=(pred-labels)[mask].square().mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(latent.parameters())+list(condition.parameters()),2);opt.step()
        if step==0 or (step+1)%20==0:
            trace.append(dict(step=step+1,loss=float(loss.detach())));print(json.dumps(dict(stage='latent',**trace[-1])),flush=True)
    latent.eval();condition.eval();states=[];inner_states=[];sse=0.;count=0;inner_sse=0.;inner_count=0;masked_sse=0.;masked_count=0
    with torch.no_grad():
        for pack,dest in ((train,states),(valid,inner_states)):
            for i in range(len(pack['hours'])):
                z,pred,labels,mask=get(pack,i,False);dest.append(z.cpu().numpy())
                if pack is train:sse+=float((pred-labels)[mask].square().sum())*scale**2;count+=int(mask.sum())
                else:inner_sse+=float((pred-labels)[mask].square().sum())*scale**2;inner_count+=int(mask.sum())
                # Query/basis features are independent of label availability.
                # Build the same Phi by differentiating its linear readout:
                # directly re-use get's condition calculation with a label
                # subset through the explicit masked encoding below.
                half=mask.clone();half[:,::2]=False;held=mask&~half
                if half.any(-1).all() and held.any():
                    zhalf=latent.encode(labels,st,half)
                    h=condition.encoder(tensor(pack['history'][i]),ts)
                    xy=tensor((pack['query_xy']-pack['support_xy'].min(0))/(pack['support_xy'].max(0)-pack['support_xy'].min(0))*2-1)
                    q=F.grid_sample(h,xy[None,None].expand(6,-1,-1,-1),align_corners=True,padding_mode='border')[:,:,0].transpose(1,2)
                    features=torch.cat([st[None].expand(6,-1,-1)[:,:,:8],q,tensor(pack['mu'][i])[...,None]/100],-1)
                    partial=latent.decode(zhalf,features)
                    masked_sse+=float((partial-labels)[held].square().sum())*scale**2;masked_count+=int(held.sum())
    torch.save(dict(latent=latent.state_dict(),condition=condition.state_dict()),folder/'latent.pt')
    np.savez_compressed(folder/'latent.npz',train=np.stack(states),inner=np.stack(inner_states))
    write_json(folder/'latent_metrics.json',dict(reconstruction_train_mm=(sse/count)**.5,
        reconstruction_inner2019_mm=(inner_sse/inner_count)**.5,
        masked_label_reconstruction_mm=(masked_sse/masked_count)**.5,masked_events=masked_count,
        steps=args.representation_steps,trace=trace))
    write_json(folder/'contract.json',dict(bias_mm=bias,scale_mm=scale,anchor_sha256=sha(folder/'anchor.npz'),
        latent_weights_sha256=sha(folder/'latent.pt'),stats_role='2016-2018 OOF training only',
        dense_reconstruction_claim=False,anchor_gate_passed=anchor_metrics['reconstruction_train_mm']<=1))
    return anchor_metrics


def probability_run(root,train,valid,args,arm):
    folder=root/arm;folder.mkdir(exist_ok=False);rep=root/'representation'
    anchor=dict(np.load(rep/'anchor.npz'));latent=dict(np.load(rep/'latent.npz'))
    is_latent=arm.startswith('L');is_future=arm.startswith('J');spectral=arm=='L1'
    states=latent['train'] if is_latent else anchor['train'];test=latent['inner'] if is_latent else anchor['inner']
    if is_future:
        # Separate F labels and forecast-origin conditions. No R future mu.
        ftrain=load_events(root/'oof','future_train');ftest=load_events(root/'oof','future_inner')
        H=anchor['H'];field=AnchorField(anchor['xy'])
        states=field.encode((ftrain['residual']-anchor['bias'])/anchor['scale'],H)
        test=field.encode((ftest['residual']-anchor['bias'])/anchor['scale'],H)
        train,valid=ftrain,ftest
        train_error=(states@H.T-(train['residual']-anchor['bias'])/anchor['scale'])*anchor['scale']
        inner_error=(test@H.T-(valid['residual']-anchor['bias'])/anchor['scale'])*anchor['scale']
        missing=(train['residual']-anchor['bias'])/anchor['scale'];missing=missing.copy();missing[:,:,::2]=np.nan
        partial=field.encode(missing,H)@H.T
        masked_error=(partial-(train['residual']-anchor['bias'])/anchor['scale'])*anchor['scale'];masked_error[:,:,1::2]=np.nan
        gate_metrics=dict(reconstruction_train_mm=float(np.sqrt(np.nanmean(train_error**2))),
            reconstruction_inner2019_mm=float(np.sqrt(np.nanmean(inner_error**2))),
            masked_label_reconstruction_mm=float(np.sqrt(np.nanmean(masked_error**2))),masked_events=int(np.isfinite(masked_error).sum()))
    else:gate_metrics=json.loads((rep/('latent_metrics.json' if is_latent else 'anchor_metrics.json')).read_text())
    gate=check_family_gate(gate_metrics,args.diagnostic);write_json(folder/'eligibility.json',gate)
    transform=InnovationTransform.fit(states);v=transform.forward(states)
    U=anchor['U'];band=arm=='B1';joint=arm=='J1'
    if is_future:
        if joint:
            matrix=np.stack([joint_transform(np.eye(6*256)[i].reshape(6,256),U).ravel() for i in range(6*256)])
        else:matrix=np.eye(6*256)
        v=(v.reshape(len(v),-1)@matrix).reshape(len(v),1,-1)
    band_bias=np.zeros(v.shape[-1]);band_scale=np.ones(v.shape[-1])
    if band:
        frequency=v@U;band_bias=frequency.mean((0,1));band_scale=np.maximum(frequency.std((0,1)),1e-4)
        v=(frequency-band_bias)/band_scale
    dimension=v.shape[-1];torch.manual_seed(args.seed)
    condition=ProbabilityCondition(spectral=spectral).cuda()
    domain='graph' if arm=='A1' else 'dual' if arm in ('A2','B0','B1') else 'spatial'
    effectiveU=U if dimension==256 else np.eye(dimension)
    M=np.diag(band_scale)@U.T if band else np.eye(dimension)
    models={'diffusion':InnovationDiffusion(dimension,condition.output_dim,domain,effectiveU,state_to_space=M).cuda(),
            'gaussian':ConditionalGaussian(dimension,condition.output_dim).cuda()}
    # Both controls train the same G from their own loss, with identical seed
    # initialization. G is not optimized once using only the diffusion target.
    results={};initial={k:x.cpu().clone() for k,x in condition.state_dict().items()}
    for family,model in models.items():
        torch.manual_seed(args.seed);condition.load_state_dict(initial)
        opt=torch.optim.AdamW(list(condition.parameters())+list(model.parameters()),lr=3e-4,weight_decay=1e-4)
        trace=[];start=time.monotonic();ts=tensor(train['static'])
        for step in range(args.probability_steps):
            i=step%len(v);idx=0 if is_future else slice(None)
            h=tensor(train['history'][i,idx]);clock=tensor(train['clock'][i,idx]);spec=tensor(train['spectrum'][i,idx])
            if is_future:h=h[None];clock=clock[None];spec=spec[None]
            c=condition(h,ts,clock,spec);loss=model.loss(tensor(v[i]),c)
            opt.zero_grad();loss.backward();grad=nn.utils.clip_grad_norm_(list(condition.parameters())+list(model.parameters()),2.)
            if not torch.isfinite(grad):raise FloatingPointError('probability gradient nonfinite')
            opt.step()
            if step==0 or (step+1)%20==0:
                row=dict(step=step+1,loss=float(loss.detach()),seconds=time.monotonic()-start);trace.append(row)
                print(json.dumps(dict(stage=arm,family=family,**row)),flush=True)
        torch.save(dict(condition=condition.state_dict(),model=model.state_dict()),folder/f'{family}.pt')
        condition.eval();model.eval();generated=[];rng=torch.Generator(device='cuda').manual_seed(args.seed+300)
        with torch.no_grad():
            for i in range(len(valid['hours'])):
                idx=0 if is_future else slice(None)
                h=tensor(valid['history'][i,idx]);clock=tensor(valid['clock'][i,idx]);spec=tensor(valid['spectrum'][i,idx])
                if is_future:h=h[None];clock=clock[None];spec=spec[None]
                c=condition(h,ts,clock,spec);n=len(c)
                noise=torch.randn(args.members,n,dimension,device='cuda',generator=rng)
                samples=model.sample(c.repeat(args.members,1),noise.reshape(-1,dimension),steps=args.ddim_steps).cpu().numpy().reshape(args.members,n,dimension)
                if band:samples=(samples*band_scale+band_bias)@U.T
                if is_future:samples=(samples[:,0]@matrix.T).reshape(args.members,6,256)
                state=transform.inverse(samples)
                if is_latent:
                    saved=torch.load(rep/'latent.pt',weights_only=True);decoder=LatentField(condition_width=63).cuda();decoder.load_state_dict(saved['latent'])
                    oldG=ProbabilityCondition().cuda();oldG.load_state_dict(saved['condition']);oldG.eval()
                    hq=oldG.encoder(tensor(valid['history'][i]),ts)
                    xy=tensor((valid['query_xy']-valid['support_xy'].min(0))/(valid['support_xy'].max(0)-valid['support_xy'].min(0))*2-1)
                    q=F.grid_sample(hq,xy[None,None].expand(6,-1,-1,-1),align_corners=True,padding_mode='border')[:,:,0].transpose(1,2)
                    static=tensor(valid['query_static'][:,:8])[None].expand(6,-1,-1)
                    features=torch.cat([static,q,tensor(valid['mu'][i])[...,None]/100],-1)
                    u=decoder.decode(tensor(state.reshape(-1,128)),features.repeat(args.members,1,1)).cpu().numpy().reshape(args.members,6,410)
                else:u=state@anchor['H'].T
                generated.append(valid['mu'][i][None]+anchor['bias']+anchor['scale']*u)
        samples=np.stack(generated,1);metrics=scores(samples,valid['truth'])
        np.savez_compressed(folder/f'{family}_evaluation.npz',samples=samples.astype('float32'),truth=valid['truth'],hours=valid['hours'])
        results[family]=dict(metrics=metrics,trace=trace,steps=args.probability_steps,parameters=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in condition.parameters()))
        write_json(folder/f'{family}_fit.json',results[family])
    np.savez_compressed(folder/'transforms.npz',**transform.arrays(),band_bias=band_bias,band_scale=band_scale)
    write_json(folder/'contract.json',dict(arm=arm,domain=domain,dimension=dimension,forecast=is_future,
        diagnostic=args.diagnostic,seed=args.seed,normalization='training OOF only',
        standard_innovation_chain=True,shared_noise_state=True,stats_sha256=sha(folder/'transforms.npz'),
        task='F six-hour origin-persistence mean' if is_future else 'R current-hour reconstruction'))
    write_json(folder/'results.json',results)
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=PROJECT/'results/20261125_temporal_continuous_data')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['all','foundation','oof','representation','probability'],default='all')
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--oof-steps',type=int,default=3000)
    p.add_argument('--representation-steps',type=int,default=2000);p.add_argument('--probability-steps',type=int,default=3000)
    p.add_argument('--train-hours',type=int,default=720);p.add_argument('--eval-hours',type=int,default=72)
    p.add_argument('--members',type=int,default=64);p.add_argument('--ddim-steps',type=int,default=20)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--kinds',nargs='+',default=['T0','T1','T2','T3','T4','T5'])
    p.add_argument('--arms',nargs='+',default=['L0','L1','A0','A1','A2','B0','B1','J0','J1'])
    p.add_argument('--diagnostic',action='store_true');args=p.parse_args()
    if args.train_hours%6 or args.eval_hours%6:raise ValueError('hours must be multiples of six')
    args.out.mkdir(parents=True,exist_ok=True);root=args.out;torch.set_num_threads(4)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    hashes={f:sha(Path(__file__).with_name(f)) for f in ('joint_experiment.py','joint_data.py','joint_probability.py','spectral_operators.py','masked_transformer_fusion.py')}
    data_hash=sha(args.data/'contract.json');verify_lineage(root,args.diagnostic,data_hash,hashes)
    receipt=root/f'protocol_{args.phase}.json'
    if receipt.exists():
        prior=json.loads(receipt.read_text())
        if prior['config']!=config:raise ValueError('phase arguments differ; use a new output')
    else:write_json(receipt,dict(config=config,data_sha256=data_hash,
        code_sha256=hashes,
        status='diagnostic pilot' if args.diagnostic else 'preregistered fixed-step candidate',
        confirmation_C_D_read=False,training_time='continuous chronological hours, station identities only sampled',
        inner='2016-2018 fit /2019 diagnostic, no2020 selection',forecast_mean='origin-persistence control, not trained F foundation'))
    if args.phase in ('all','foundation'):
        summary=json.loads((root/'foundation_summary.json').read_text()) if (root/'foundation_summary.json').exists() else []
        for kind in args.kinds:
            folder=root/'foundation'/kind
            if (folder/'monitor.json').exists():continue
            data=JointData(args.data,side=8 if kind=='T0' else 16)
            model=train_foundation(data,kind,args.steps,args.seed,folder)
            m=foundation_monitor(data,model,args.eval_hours);write_json(folder/'monitor.json',m);summary.append(dict(kind=kind,**m))
            del data,model;torch.cuda.empty_cache()
        write_json(root/'foundation_summary.json',summary)
    if args.phase in ('all','oof'):
        groups=np.load(args.data/'geometry.npz')['groups']
        for f,outer in enumerate(fold_groups(groups)):
            folder=root/'oof'/f'fold{f}'
            if (folder/'future_inner_receipt.json').exists():continue
            keep=np.setdiff1d(np.arange(410),outer);data=JointData(args.data,train_ids=keep)
            model=train_foundation(data,'T5',args.oof_steps,args.seed,folder)
            # Keep enough pre-roll for 72 causal means, each with72 raw hours.
            for tag,year,hours in [('train',2016,args.train_hours),('inner',2019,args.eval_hours)]:
                data.set_split(year,year+1);start=data.split_start+7*24
                collect_fold(data,model,outer,start,hours,folder,tag)
                collect_fold(data,model,outer,start,hours,folder,'future_'+tag,forecast=True)
            del data,model;torch.cuda.empty_cache()
    if args.phase in ('all','representation','probability'):
        train=load_events(root/'oof','train');inner=load_events(root/'oof','inner')
        if args.phase!='probability':representation(root,train,inner,args)
        metrics=json.loads((root/'representation/anchor_metrics.json').read_text())
        # Eligibility is checked per family inside probability_run; L and F
        # never inherit the R-anchor family's self-reconstruction result.
    if args.phase in ('all','probability'):
        summary=json.loads((root/'probability_summary.json').read_text()) if (root/'probability_summary.json').exists() else {}
        for arm in args.arms:
            if (root/arm/'results.json').exists():continue
            summary[arm]=probability_run(root,train,inner,args,arm);write_json(root/'probability_summary.json',summary)
    write_json(root/'status.json',dict(status='COMPLETE_'+args.phase.upper(),diagnostic=args.diagnostic,
        stage='local engineering pilot' if args.diagnostic else 'fixed-step candidate',confirmation_C_D_read=False))


if __name__=='__main__':main()
