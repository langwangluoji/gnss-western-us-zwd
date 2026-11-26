"""End-to-end graph fusion with masked-station pretraining and latent sequences."""
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist
import graph_fusion_spike as old
from gnss_field.artifacts import write_json,sha256

core=old.core
ROOT=core.PROJECT/'results/20261123_masked_graph_v06'


def geometry(a):
    g=old.geometry(a)
    nodes=np.unique(old.QI);ax=a['xy'][a['aids']];ah=a['heights'][a['aids']]
    qn=g['qn'].cpu().numpy()
    edge=np.concatenate([(ax[qn]-old.XY[nodes,None])/150000,
        (ah[qn]-old.TERRAIN[nodes,0,None]*3000)[...,None]/750],-1)
    g['qedge']=torch.tensor(edge,dtype=torch.float32,device='cuda')
    return g


def conditions(raw,hidden=None):
    valid=torch.isfinite(raw).all(-1)
    if hidden is not None:valid=valid&~hidden[None,None]
    safe=torch.where(valid[...,None],raw,0.)
    # Fixed constants, no full-network normalization that includes hidden targets.
    offset=raw.new_tensor([100.,280.,900.]);scale=raw.new_tensor([50.,10.,80.])
    normalized=torch.where(valid[...,None],(safe-offset)/scale,0.)
    avg=safe[...,0].sum(-1)/valid.sum(-1).clamp_min(1)
    innovation=torch.where(valid,(safe[...,0]-avg[...,None])/50,0.)
    block=torch.cat([normalized,innovation[...,None],valid[...,None]],-1)
    x=torch.cat([block[:,6-lag:30-lag] for lag in (0,1,3)],-1)
    return x,valid[:,6:],safe[:,6:]


class GraphField(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(15,32),nn.SiLU())
        self.key=nn.Linear(32,32,bias=False);self.edge=nn.Linear(3,32,bias=False)
        self.score=nn.Linear(32,1,bias=False);self.value=nn.Linear(32,32);self.norm=nn.LayerNorm(32)
        self.qscore=nn.Sequential(nn.Linear(35,24),nn.SiLU(),nn.Linear(24,1))
        self.prior=nn.Sequential(nn.Linear(13,32),nn.SiLU(),nn.Linear(32,1))
        self.obs=nn.Sequential(nn.Linear(45,32),nn.SiLU(),nn.Linear(32,1))
        self.gate=nn.Sequential(nn.Linear(45,1),nn.Sigmoid())
        for head in (self.prior,self.obs):nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)

    def forward(self,raw,cycles,g,nodes,hidden=None):
        x,valid,safe=conditions(raw,hidden);h=self.enc(x);nb=g['neighbors'];nh=h[:,:,nb]
        logits=self.score(torch.tanh(self.key(h)[:,:,:,None]+self.key(nh)+self.edge(g['edges']))).squeeze(-1)
        available=valid[:,:,nb];w=torch.softmax(logits.masked_fill(~available,-1e4),-1)*available
        w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
        h=self.norm(h+(w[...,None]*self.value(nh)).sum(-2))
        qn=g['qn'][nodes];hs=h[:,:,qn];ve=valid[:,:,qn]
        edge=g['qedge'][nodes][None,None].expand(*hs.shape[:-1],3)
        logits=self.qscore(torch.cat([hs,edge],-1)).squeeze(-1)+g['qweight'][nodes].clamp_min(1e-20).log()
        w=torch.softmax(logits.masked_fill(~ve,-1e4),-1)*ve;w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
        agg=(hs*w[...,None]).sum(-2)
        # Learned observation weights and baseline are both recomputed after hiding.
        avg=(safe[:,:,qn]*w[...,None]).sum(-2)
        spread=torch.sqrt(((safe[:,:,qn,0]-avg[...,0,None])**2*w).sum(-1)+1e-6)/50
        den=(g['qweight'][nodes]*ve).sum(-1)
        terrain=g['qstatic'][nodes][None,None].expand(*avg.shape[:-1],4)
        cyc=cycles[:,:,None].expand(*avg.shape[:-1],4)
        stats=torch.cat([terrain,cyc,(avg-avg.new_tensor([100.,280.,900.]))/avg.new_tensor([50.,10.,80.]),spread[...,None],torch.log1p(den)[...,None]],-1)
        features=torch.cat([agg,stats],-1);gate=self.gate(features)
        delta=10*(gate*self.prior(stats)+(1-gate)*self.obs(features)).squeeze(-1)
        return avg[...,0]+delta,features


class LatentField(core.RandomField):
    def __init__(self,correlated):
        super().__init__(correlated)
        self.head=nn.Sequential(nn.Linear(45,32),nn.SiLU(),nn.Linear(32,16))


def dataset(a,name):
    hours=a[name+'_hours'];day=hours/24
    cycles=np.stack([np.sin(hours*2*np.pi/24),np.cos(hours*2*np.pi/24),np.sin(day*2*np.pi/365.25),np.cos(day*2*np.pi/365.25)],-1)[:,6:].astype('f4')
    # Reuse the locked support population, never shrink based on model error.
    _,_,support=core.encode_inputs(a,a[name+'_raw'],hours,a['xy'],a['terrain'])
    y=a[name+'_y'];mask=np.isfinite(y)&support&a['qvalid']
    return {k:torch.as_tensor(v,device='cuda') for k,v in dict(raw=a[name+'_raw'],cycles=cycles,
        y=np.nan_to_num(y[:,6:]),mask=mask[:,6:],height=np.exp(-a['heights']/3000).astype('f4')).items()}


def predict(model,d,g,rows,ids,hidden=None,features=False):
    nodes,rev=torch.unique(g['index'][ids].flatten(),return_inverse=True)
    z,h=model(d['raw'][rows],d['cycles'][rows],g,nodes,hidden)
    index=rev.reshape(-1,4);w=g['weights'][ids]
    p=(z[:,:,index]*w).sum(-1)*d['height'][ids]
    if features:return p,h[:,:,index],w
    return p


@torch.no_grad()
def mean_eval(model,d,g,ids):
    p=torch.cat([predict(model,d,g,[i],ids) for i in range(len(d['raw']))])
    return old.inv.point_score(p.cpu().numpy(),d['y'][:,:,ids].cpu().numpy(),d['mask'][:,:,ids].cpu().numpy()),p


def fit_mean(ds,a,g,seed,arm):
    torch.manual_seed(seed);m=GraphField().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001)
    rng=np.random.default_rng(seed);d=ds['train'];trace=[];fit=a['split_fit'];iid=a['split_inner']
    for step in range(1,1601):
        if step==401:
            opt=torch.optim.AdamW(m.parameters(),lr=.001);rng=np.random.default_rng(seed+100)
            score,_=mean_eval(m,ds['inner'],g,iid);best=score['macro_rmse'];state=copy.deepcopy(m.state_dict());selected=400
            trace.append(dict(step=400,inner=score))
        row=[int(rng.integers(len(d['raw'])))];hidden=None
        if arm=='masked' and step<=400:
            loc=rng.choice(len(a['aids']),44,replace=False);ids=a['aids'][loc]
            hidden=torch.zeros(len(a['aids']),dtype=torch.bool,device='cuda');hidden[loc]=True
            raw=d['raw'][row][:,6:,loc,0];mask=torch.isfinite(raw)&torch.tensor(a['qvalid'][ids],device='cuda')
            y=torch.nan_to_num(raw)*d['height'][ids]
        else:
            ids=rng.choice(fit,32,replace=False);y=d['y'][row][:,:,ids];mask=d['mask'][row][:,:,ids]
        p=predict(m,d,g,row,ids,hidden);loss=(((p-y)/10)**2*mask).sum()/mask.sum().clamp_min(1)
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
        if step>400 and step%200==0:
            score,_=mean_eval(m,ds['inner'],g,iid);trace.append(dict(step=step,inner=score,loss=float(loss.detach())))
            if score['macro_rmse']<best:best=score['macro_rmse'];state=copy.deepcopy(m.state_dict());selected=step
            print(json.dumps(dict(stage='mean',seed=seed,arm=arm,step=step,best=best)),flush=True)
    m.load_state_dict(state);torch.save(dict(model=state,selected_step=selected),ROOT/f'{arm}_{seed}.pt')
    return m,dict(trace=trace,selected_step=selected,inner_macro_rmse=best)


@torch.no_grad()
def cache(m,d,g,ids):
    chunks=[predict(m,d,g,[i],ids,features=True) for i in range(len(d['raw']))]
    return dict(mu=torch.cat([v[0] for v in chunks]),h=torch.cat([v[1] for v in chunks]),w=chunks[0][2],
                height=d['height'][ids],y=d['y'][:,:,ids],mask=d['mask'][:,:,ids])


def samples(gen,d,noise):
    b,t,n,c,f=d['h'].shape
    residual=gen(d['h'].reshape(b,t,n*c,f),noise).reshape(b,noise.shape[1],t,n,c)
    return d['mu'][:,None]+(residual*d['w'][None,None,None]).sum(-1)*d['height']


@torch.no_grad()
def prob_eval(gen,d,k=64,seed=9301):
    torch.manual_seed(seed);s=samples(gen,d,torch.randn(len(d['h']),k,24,8,device='cuda'))
    return old.inv.probability_score(s.cpu().numpy(),d['y'].cpu().numpy(),d['mask'].cpu().numpy()),s


def fit_gen(ds,seed,arm,correlated):
    torch.manual_seed(seed);gen=LatentField(correlated).cuda()
    opt=torch.optim.AdamW([{'params':gen.head.parameters(),'lr':.002},{'params':[gen.rho_raw],'lr':.02}])
    rng=np.random.default_rng(seed);best=float('inf');trace=[];d=ds['train']
    for step in range(1,401):
        row=[int(rng.integers(len(d['h'])))];sub={k:v[row] if k in ('h','mu','y','mask') else v for k,v in d.items()}
        s=samples(gen,sub,torch.randn(1,8,24,8,device='cuda'));loss=core.objective(s,sub['y'],sub['mask'])
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(gen.parameters(),2);opt.step()
        if step%100==0:
            with torch.random.fork_rng(devices=[0]):score,_=prob_eval(gen,ds['inner'],32)
            criterion=score['crps']+.5*score['increment_crps'];trace.append(dict(step=step,inner=score))
            if criterion<best:best=criterion;state=copy.deepcopy(gen.state_dict());selected=step
    kind='ar' if correlated else 'ind';key=f'{arm}_{seed}_{kind}';gen.load_state_dict(state)
    torch.save(dict(model=state,selected_step=selected),ROOT/f'{key}.pt')
    print(json.dumps(dict(stage='latent',key=key,selected=selected,inner=best)),flush=True)
    return dict(trace=trace,selected_step=selected,rho=float(.98*gen.rho_raw.tanh()) if correlated else 0.)


def add_reserved(a):
    hours=np.array([int(np.datetime64(f'2018-{m:02d}-27','h').astype('i8'))+np.arange(-6,24) for m in range(1,13)])
    store=core.PackedStore(core.DATA,purpose='train');a['reserved_hours']=hours
    a['reserved_raw']=store.context_values(hours.ravel())[...,:3].reshape(12,30,len(a['aids']),3)
    a['reserved_y']=np.full((12,30,len(a['heights'])),np.nan,dtype='f4')
    a['reserved_y'][...,a['bids']]=store.targets('B',hours.ravel()).reshape(12,30,len(a['bids']))


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(2);start=time.monotonic()
    a=dict(np.load(old.exp.ROOT/'data.npz'));source=json.loads((old.exp.ROOT/'results.json').read_text())
    assert sha256(old.exp.ROOT/'data.npz')==source['runtime']['data_sha256']
    write_json(ROOT/'experiment.lock.json',dict(code_sha256=sha256(__file__),protocol_sha256=sha256(ROOT/'PROTOCOL.md'),data_sha256=source['runtime']['data_sha256'],
       dependencies={n:sha256(core.PROJECT/'tools'/n) for n in ['graph_fusion_spike.py','sequence_innovation_spike.py','explore_sequence_mechanisms.py','sequence_grid_spike.py','sequence_mechanism_spike.py']},C_D_labels_read=False))
    g=geometry(a);ds={n:dataset(a,n) for n in ('train','inner')};selection={}
    for seed in (42,17):
        for arm in ('direct','masked'):
            model,selection[f'{arm}_{seed}']=fit_mean(ds,a,g,seed,arm)
            cached={n:cache(model,ds[n],g,a['split_fit'] if n=='train' else a['split_inner']) for n in ds}
            for corr in ([True] if arm=='direct' else [False,True]):
                selection[f'{arm}_{seed}_'+('ar' if corr else 'ind')]=fit_gen(cached,seed,arm,corr)
            del cached
    write_json(ROOT/'selection.lock.json',selection);del ds
    add_reserved(a);results={}
    for name in ('confirmation','reserved'):
        d=dataset(a,name);ids=a['split_outer']
        for seed in (42,17):
            for arm in ('direct','masked'):
                model=GraphField().cuda();model.load_state_dict(torch.load(ROOT/f'{arm}_{seed}.pt',weights_only=False)['model'])
                point,p=mean_eval(model,d,g,ids);cached=cache(model,d,g,ids)
                ref=np.load(old.inv.ROOT/f'history_{seed}_{name}.npz')
                assert np.array_equal(ref['mask'],cached['mask'].cpu().numpy())
                for corr in ([True] if arm=='direct' else [False,True]):
                    key=f'{arm}_{seed}_'+('ar' if corr else 'ind');gen=LatentField(corr).cuda()
                    gen.load_state_dict(torch.load(ROOT/f'{key}.pt',weights_only=False)['model'])
                    score,s=prob_eval(gen,cached);check,_=prob_eval(gen,cached,128,11703)
                    results[key+'_'+name]=dict(point=point,k64=score,k128=check)
                    np.savez_compressed(ROOT/f'{key}_{name}.npz',prediction=p.cpu().numpy(),samples=s.cpu().numpy(),target=ref['target'],mask=ref['mask'],station_ids=ids,hours=ref['hours'])
    write_json(ROOT/'results.json',dict(results=results,selection=selection,runtime=dict(seconds=time.monotonic()-start,device=torch.cuda.get_device_name(),code_sha256=sha256(__file__),
       checkpoints={p.name:sha256(p) for p in ROOT.glob('*.pt')},selection_sha256=sha256(ROOT/'selection.lock.json'))))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
