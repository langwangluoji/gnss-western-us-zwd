"""Throwaway real-GNSS sequence experiment; see frozen PROTOCOL.md.

Uses no PWV/Tm values, no pretrained weights, and no C/D labels.
Historical packed cohort selection DID use those variables upstream.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.ndimage import uniform_filter
from scipy.spatial.distance import cdist
from simple_weather_unet_pilot import PROJECT, DATA, spatial_split, query_table, metrics
from gnss_field.data import PackedStore
from gnss_field.artifacts import sha256, write_json

ROOT = PROJECT/'results/20261123_sequence_spike_v01'
DEVICE = 'cuda'


def calendars():
    out = {}
    for name, years, days in [('train', [2016, 2017], [8, 18, 25]),
                              ('inner', [2018], [1]), ('outer', [2018], [16])]:
        out[name] = np.array([int(np.datetime64(f'{y}-{m:02d}-{d:02d}', 'h').astype('i8'))
                             + np.arange(-6, 24) for y in years for m in range(1, 13) for d in days])
    return out


def prepare():
    store = PackedStore(DATA, purpose='train')
    cal = calendars()
    aids, bids = store.ids('A'), store.ids('B')
    splits = spatial_split(store.stations)
    normraw = store.context_values(cal['train'].ravel())[..., :3]
    mean = np.nanmean(normraw, (0, 1)); std = np.maximum(np.nanstd(normraw, (0, 1)), 1.)
    _, qi, qw, qvalid = query_table(store)
    h = store.stations.height.to_numpy()
    gy, gx = np.gradient(store.grid['height'], 25000., 25000.)
    z = store.grid['height']
    rough = np.sqrt(np.maximum(uniform_filter(z*z, 5)-uniform_filter(z, 5)**2, 0))
    terrain = np.stack([h/3000, (gx.ravel()[qi]*qw).sum(-1)/.1,
                        (gy.ravel()[qi]*qw).sum(-1)/.1, (rough.ravel()[qi]*qw).sum(-1)/500], -1)
    archive = dict(mean=mean, std=std, heights=h, xy=store.xy, terrain=terrain,
                   qvalid=qvalid, aids=aids, bids=bids, **{f'split_{k}': v for k, v in splits.items()})
    for name, hours in cal.items():
        raw = store.context_values(hours.ravel())[..., :3].reshape(*hours.shape, len(aids), 3)
        y = np.full((*hours.shape, len(h)), np.nan, dtype='f4')
        y[..., bids] = store.targets('B', hours.ravel()).reshape(*hours.shape, len(bids))
        archive[f'{name}_raw'] = raw
        archive[f'{name}_hours'] = hours
        archive[f'{name}_y'] = y
    np.savez(ROOT/'data.npz', **archive)
    write_json(ROOT/'data_receipt.json', dict(manifest=store.manifest['fingerprint'],
        source_code=sha256(__file__), data_sha256=sha256(ROOT/'data.npz'),
        protocol_sha256=sha256(ROOT/'PROTOCOL.md'), input_columns=[0, 1, 2],
        input_names=['A_GNSS_Z0', 'temperature_k', 'pressure_hpa'], pretrained=False,
        cohort='historical PWV-completeness filtered', C_D_labels_read=False,
        sequence_counts={k: len(v) for k, v in cal.items()}, hours_per_sequence=30,
        scored_hours=24, splits={k: v.tolist() for k,v in splits.items()},
        normalisation_role='A training sequence hours', mean=mean.tolist(), std=std.tolist()))
    return archive, store


def encode_inputs(a, raw, hours, xy, terrain):
    """One local geometric encoder for station and arbitrary grid queries."""
    dist = cdist(xy, a['xy'][a['aids']])/1000
    valid = np.isfinite(raw).all(-1)
    normalized = np.nan_to_num((raw-a['mean'])/a['std'])
    values, bases = [], []
    for radius in (50, 150, 300):
        w = np.exp(-.5*(dist/radius)**2)*(dist <= 3*radius)
        den = np.einsum('qa,bta->btq', w, valid)+1e-8
        avg = np.einsum('qa,btav->btqv', w, normalized*valid[..., None])/den[..., None]
        second = np.einsum('qa,bta->btq', w, normalized[..., 0]**2*valid)/den
        values.extend([avg, np.sqrt(np.maximum(second-avg[..., 0]**2, 0))[..., None],
                       np.log1p(den)[..., None]])
        bases.append(avg[..., 0]*a['std'][0]+a['mean'][0])
    static = np.c_[xy/1e6, terrain, dist.min(-1)/100]
    times = hours.astype('datetime64[h]')
    day = (times-times.astype('datetime64[Y]')).astype('timedelta64[h]').astype(float)/(24*365.25)
    cyc = np.stack([np.sin(hours%24/24*2*np.pi), np.cos(hours%24/24*2*np.pi),
                    np.sin(day*2*np.pi), np.cos(day*2*np.pi)], -1)
    shape = (*raw.shape[:2], len(xy))
    values.extend([np.broadcast_to(static, (*shape, static.shape[-1])),
                   np.broadcast_to(cyc[:, :, None], (*shape, 4))])
    x = np.concatenate(values, -1).astype('f4')
    support = np.einsum('qa,bta->btq', (dist <= 150), valid.astype(int)) >= 3
    return x, bases[0].astype('f4'), support


class MeanModel(nn.Module):
    def __init__(self, dim, temporal):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU())
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Linear(32, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.temporal = temporal

    def forward(self, x):
        b, t, n, _ = x.shape
        h = self.enc(x)
        if self.temporal:
            h = self.gru(h.permute(0, 2, 1, 3).reshape(b*n, t, 32))[0]
            h = h.reshape(b, n, t, 32).permute(0, 2, 1, 3)
        return self.head(h)[..., 0]*10, h


class RandomField(nn.Module):
    def __init__(self, correlated):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 16))
        self.rho_raw = nn.Parameter(torch.tensor(0.))
        self.correlated = correlated

    def forward(self, h, noise):
        # noise B,K,T,8. Same draw is shared over all spatial queries.
        rho = .98*torch.tanh(self.rho_raw) if self.correlated else self.rho_raw*0
        latent = [noise[:, :, 0]]
        for t in range(1, noise.shape[2]):
            latent.append(rho*latent[-1]+torch.sqrt(1-rho*rho)*noise[:, :, t])
        z = torch.stack(latent, 2)
        basis = torch.cat([z, (z*z-1)/np.sqrt(2)], -1)
        coef = self.head(h)*3
        return torch.einsum('btnr,bktr->bktn', coef, basis)


def fair_crps(samples, y, mask):
    # B,K,T,N, finite labels, fixed masks; K(K-1) U-statistic.
    k = samples.shape[1]
    first = (samples-y[:, None]).abs().mean(1)
    sorted_x = samples.sort(dim=1).values
    coeff = (2*torch.arange(k, device=samples.device)-k+1).reshape(1,k,1,1)
    second = (sorted_x*coeff).sum(1)/(k*(k-1))
    return ((first-second)*mask).sum()/mask.sum().clamp(min=1)


def objective(s, y, m):
    marginal = fair_crps(s, y, m)
    temporal = fair_crps(s[:,:,1:]-s[:,:,:-1], y[:,1:]-y[:,:-1], m[:,1:]&m[:,:-1])
    # Fixed adjacent sites in locked station order, not nearest neighbours.
    spatial = fair_crps(s[:,:,:,1:]-s[:,:,:,:-1], y[:,:,1:]-y[:,:,:-1], m[:,:,1:]&m[:,:,:-1])
    return (marginal+.5*temporal+.1*spatial)/10


def mean_eval(model, data, ids):
    with torch.no_grad():
        p = (model(data['x'][:,:,ids])[0]+data['base'][:,:,ids])*data['height'][ids]
    return metrics(p[:,6:].cpu().numpy().reshape(-1,len(ids)),
                   data['y'][:,6:,ids].cpu().numpy().reshape(-1,len(ids)),
                   data['mask'][:,6:,ids].cpu().numpy().reshape(-1,len(ids)))


def train_mean(ds, ids, inner_ids, seed, temporal, steps):
    torch.manual_seed(seed)
    model = MeanModel(ds['train']['x'].shape[-1], temporal).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    rng = np.random.default_rng(seed); history=[]; best=float('inf'); state=None
    for step in range(1,steps+1):
        rows = rng.integers(len(ds['train']['x']), size=4)
        d = ds['train']; x=d['x'][rows][:,:,ids]
        p=(model(x)[0]+d['base'][rows][:,:,ids])*d['height'][ids]
        m=d['mask'][rows][:,6:,ids]; y=d['y'][rows][:,6:,ids]
        loss=(((p[:,6:]-y)/10)**2*m).sum()/m.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2); opt.step()
        if step%100==0 or step==steps:
            score=mean_eval(model,ds['inner'],inner_ids)
            history.append(dict(step=step,loss=float(loss.detach()),inner=score))
            if score['macro_rmse']<best: best=score['macro_rmse']; state=copy.deepcopy(model.state_dict())
            print(json.dumps(dict(stage='mean',seed=seed,temporal=temporal,**history[-1])),flush=True)
    model.load_state_dict(state); model.eval()
    return model,history


@torch.no_grad()
def sample_eval(gen, d, ids, seed=9301, members=64):
    torch.manual_seed(seed)
    noise=torch.randn(len(d['h']),members,d['h'].shape[1],8,device=DEVICE)
    s=(d['mu'][:,:,ids][:,None]+gen(d['h'][:,:,ids],noise))*d['height'][ids]
    s=s[:,:,6:]; y=d['y'][:,6:,ids]; m=d['mask'][:,6:,ids]
    low,high=torch.quantile(s,torch.tensor([.05,.95],device=DEVICE),dim=1)
    pred=s.mean(1)
    result=metrics(pred.cpu().numpy().reshape(-1,len(ids)),y.cpu().numpy().reshape(-1,len(ids)),m.cpu().numpy().reshape(-1,len(ids)))
    result.update(crps=float(fair_crps(s,y,m)),increment_crps=float(fair_crps(s[:,:,1:]-s[:,:,:-1],y[:,1:]-y[:,:-1],m[:,1:]&m[:,:-1])),
        coverage90=float((((y>=low)&(y<=high))&m).sum()/m.sum()),width90=float(((high-low)*m).sum()/m.sum()),
        rho=float(.98*torch.tanh(gen.rho_raw)) if gen.correlated else 0.,members=members)
    per_sequence=[]
    for b in range(len(y)):
        per_sequence.append(dict(crps=float(fair_crps(s[b:b+1],y[b:b+1],m[b:b+1])),
          increment_crps=float(fair_crps(s[b:b+1,:,1:]-s[b:b+1,:,:-1],y[b:b+1,1:]-y[b:b+1,:-1],m[b:b+1,1:]&m[b:b+1,:-1]))))
    return result,dict(samples=s.cpu().numpy(),target=y.cpu().numpy(),mask=m.cpu().numpy()),per_sequence


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--mean-steps',type=int,default=500); parser.add_argument('--random-steps',type=int,default=350)
    args=parser.parse_args(); ROOT.mkdir(exist_ok=True); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True); started=time.monotonic()
    a,store=prepare(); ds={}
    for name in ('train','inner','outer'):
        x,base,support=encode_inputs(a,a[f'{name}_raw'],a[f'{name}_hours'],a['xy'],a['terrain'])
        y=a[f'{name}_y']; mask=np.isfinite(y)&support&a['qvalid']
        ds[name]={k:torch.as_tensor(v,device=DEVICE) for k,v in dict(x=x,base=base,
                    y=np.nan_to_num(y),mask=mask,height=np.exp(-a['heights']/3000).astype('f4')).items()}
    fit=a['split_fit']; inner=a['split_inner']; outer=a['split_outer']; results={}
    for seed in (42,17):
        result={}
        for temporal in (False,True):
            model,history=train_mean(ds,fit,inner,seed,temporal,args.mean_steps)
            name='gru' if temporal else 'frame'
            result[name]=dict(outer=mean_eval(model,ds['outer'],outer),history=history,
                              parameters=sum(p.numel() for p in model.parameters()))
            torch.save(model.state_dict(),ROOT/f'{name}_{seed}.pt')
        with torch.no_grad():
            for d in ds.values():
                # Bound cuDNN GRU inference workspace for the four-corner grid adapter.
                cached=[model(d['x'][i:i+4]) for i in range(0,len(d['x']),4)]
                residual=torch.cat([v[0] for v in cached]); h=torch.cat([v[1] for v in cached])
                d['mu']=d['base']+residual; d['h']=h.detach()
        for correlated in (False,True):
            torch.manual_seed(seed); gen=RandomField(correlated).to(DEVICE)
            opt=torch.optim.AdamW([{'params':gen.head.parameters(),'lr':.002},{'params':[gen.rho_raw],'lr':.02}])
            rng=np.random.default_rng(seed); best=float('inf'); history=[]
            for step in range(1,args.random_steps+1):
                rows=rng.integers(len(ds['train']['h']),size=4); d=ds['train']
                noise=torch.randn(4,8,30,8,device=DEVICE)
                s=(d['mu'][rows][:,:,fit][:,None]+gen(d['h'][rows][:,:,fit],noise))*d['height'][fit]
                loss=objective(s[:,:,6:],d['y'][rows][:,6:,fit],d['mask'][rows][:,6:,fit])
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(gen.parameters(),2); opt.step()
                if step%175==0 or step==args.random_steps:
                    # Independent evaluation stream does not alter subsequent training draws.
                    with torch.random.fork_rng(devices=[0]):
                        score,_,_=sample_eval(gen,ds['inner'],inner,members=32)
                    criterion=score['crps']+.5*score['increment_crps']
                    history.append(dict(step=step,inner=score))
                    if criterion<best:best=criterion;state=copy.deepcopy(gen.state_dict())
                    print(json.dumps(dict(stage='random',seed=seed,correlated=correlated,**history[-1])),flush=True)
            gen.load_state_dict(state); name='ar' if correlated else 'independent'
            score,pred,perseq=sample_eval(gen,ds['outer'],outer)
            check,_,_=sample_eval(gen,ds['outer'],outer,seed=11703,members=128)
            result[name]=dict(outer=score,independent_noise_k128=check,history=history,per_sequence=perseq)
            np.savez_compressed(ROOT/f'{name}_{seed}_predictions.npz',**pred,hours=a['outer_hours'][:,6:],station_ids=outer)
            torch.save(gen.state_dict(),ROOT/f'{name}_{seed}.pt')
        results[str(seed)]=result
        write_json(ROOT/'results.json',results)
    # Baseline on the identical population.
    d=ds['outer']; base=d['base'][:,:,outer]*d['height'][outer]
    results['baseline']=metrics(base[:,6:].cpu().numpy().reshape(-1,len(outer)),d['y'][:,6:,outer].cpu().numpy().reshape(-1,len(outer)),d['mask'][:,6:,outer].cpu().numpy().reshape(-1,len(outer)))
    results['runtime']=dict(seconds=time.monotonic()-started,device=torch.cuda.get_device_name(),mean_steps=args.mean_steps,random_steps=args.random_steps,
                            source_sha256=sha256(__file__),checkpoints={p.name:sha256(p) for p in ROOT.glob('*.pt')})
    write_json(ROOT/'results.json',results)
    print(json.dumps(dict(stage='complete',seconds=results['runtime']['seconds'])),flush=True)


if __name__=='__main__':
    main()
