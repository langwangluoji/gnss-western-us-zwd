"""Disposable paired A-graph correction on frozen generative sequences."""
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import cdist
import sequence_innovation_spike as inv
import explore_sequence_mechanisms as exp
import sequence_mechanism_spike as core
from sequence_grid_spike import XY, TERRAIN, QI, QW, GridMean, GridRandom
from gnss_field.artifacts import sha256, write_json

ROOT=core.PROJECT/'results/20261123_graph_fusion_v05'


class GraphCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(15,24),nn.SiLU())
        self.key=nn.Linear(24,24,bias=False)
        self.edge=nn.Linear(3,24,bias=False)
        self.score=nn.Linear(24,1,bias=False)
        self.value=nn.Linear(24,24)
        self.norm=nn.LayerNorm(24)
        self.head=nn.Sequential(nn.Linear(29,24),nn.SiLU(),nn.Linear(24,1))
        self.gate=nn.Sequential(nn.Linear(29,1),nn.Sigmoid())
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,x,valid,neighbors,edges,qn,qweight,qstatic):
        h=self.enc(x)
        nh=h[:,:,neighbors]
        logits=self.score(torch.tanh(self.key(h)[:,:,:,None]+self.key(nh)+self.edge(edges))).squeeze(-1)
        available=valid[:,:,neighbors]
        w=torch.softmax(logits.masked_fill(~available,-1e4),-1)*available
        w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
        h=self.norm(h+(w[...,None]*self.value(nh)).sum(-2))
        wq=qweight[None,None]*valid[:,:,qn]
        den=wq.sum(-1,keepdim=True)
        agg=(h[:,:,qn]*wq[...,None]).sum(-2)/den.clamp_min(1e-8)
        static=qstatic[None,None].expand(*agg.shape[:-1],4)
        z=torch.cat([agg,static,torch.log1p(den)],-1)
        return (10*self.head(z).squeeze(-1)*self.gate(z).squeeze(-1))


def geometry(a):
    ax=a['xy'][a['aids']];ah=a['heights'][a['aids']]
    neighbors=np.argsort(cdist(ax,ax),axis=1)[:,:12]
    edges=np.concatenate([(ax[neighbors]-ax[:,None])/150000,
                          (ah[neighbors]-ah[:,None])[...,None]/750],-1)
    nodes,rev=np.unique(QI,return_inverse=True)
    dist=cdist(XY[nodes],ax)/1000;qn=np.argsort(dist,axis=1)[:,:12]
    qw=np.exp(-.5*(np.take_along_axis(dist,qn,axis=1)/150)**2)
    return {k:torch.as_tensor(v,device='cuda') for k,v in dict(neighbors=neighbors,edges=edges.astype('f4'),
       qn=qn,qweight=qw.astype('f4'),qstatic=TERRAIN[nodes].astype('f4'),index=rev.reshape(-1,4),weights=QW.astype('f4')).items()}


def node_features(a,name,mu):
    raw=a[name+'_raw'];valid=np.isfinite(raw).all(-1)
    err=np.where(valid,raw[...,0]-mu[...,a['aids']],0)/10
    normalized=np.where(np.isfinite(raw),(raw-a['mean'])/a['std'],0)
    block=np.concatenate([normalized,err[...,None],valid[...,None]],-1)
    x=np.concatenate([block[:,6-lag:30-lag] for lag in (0,1,3)],-1)
    return x.astype('f4'),valid[:,6:]


@torch.no_grad()
def prepare(a,name,seed):
    source=json.loads((exp.ROOT/'results.json').read_text())
    path=exp.ROOT/f'distance_{seed}_2000.pt'
    assert sha256(path)==source['runtime']['checkpoints'][path.name]
    model=GridMean(27,False).cuda();model.load_state_dict(torch.load(path,weights_only=False)['model']);model.eval()
    d=exp.grid_features(a,name,False)
    mu=torch.cat([d['base'][i:i+4]+model(d['x'][i:i+4])[0] for i in range(0,len(d['x']),4)]).cpu().numpy()
    x,valid=node_features(a,name,mu)
    sel=json.loads((inv.ROOT/'selection.lock.json').read_text())[str(seed)]['history']
    height=np.exp(-a['heights']/3000)
    feat=inv.correction_features(a,name,mu)
    correction=inv.design(feat,height,np.arange(len(height)),18,np.array(sel['scales']))@np.array(sel['beta'])
    result=dict(x=x,valid=valid,base=(mu[:,6:]*height+correction).astype('f4'),
        y=d['y'][:,6:].cpu().numpy(),mask=d['mask'][:,6:].cpu().numpy(),height=height.astype('f4'))
    del d
    return {k:torch.as_tensor(v,device='cuda') for k,v in result.items()}


def predict(model,d,g,arm,rows,ids):
    neighbors=g['neighbors'];edges=g['edges']
    if arm=='self':
        neighbors=torch.arange(neighbors.shape[0],device='cuda')[:,None].expand_as(neighbors)
        edges=torch.zeros_like(edges)
    # Nonlinear model acts at grid nodes before the standard observation operator.
    nodes,reverse=torch.unique(g['index'][ids].flatten(),return_inverse=True)
    f=model(d['x'][rows],d['valid'][rows],neighbors,edges,g['qn'][nodes],g['qweight'][nodes],g['qstatic'][nodes])
    delta=(f[:,:,reverse.reshape(-1,4)]*g['weights'][ids]).sum(-1)*d['height'][ids]
    return d['base'][rows][:,:,ids]+delta


@torch.no_grad()
def evaluate(model,d,g,arm,ids):
    p=torch.cat([predict(model,d,g,arm,[i],ids) for i in range(len(d['x']))]).cpu().numpy()
    y=d['y'][:,:,ids].cpu().numpy();m=d['mask'][:,:,ids].cpu().numpy()
    return inv.point_score(p,y,m),p


def main():
    torch.set_num_threads(2);start=time.monotonic()
    a=dict(np.load(exp.ROOT/'data.npz'));source=json.loads((exp.ROOT/'results.json').read_text())
    assert sha256(exp.ROOT/'data.npz')==source['runtime']['data_sha256']
    g=geometry(a);history={};locked={}
    write_json(ROOT/'experiment.lock.json',dict(code_sha256=sha256(__file__),protocol_sha256=sha256(ROOT/'PROTOCOL.md'),
       data_sha256=source['runtime']['data_sha256'],source_checkpoints=source['runtime']['checkpoints'],
       v04_selection_sha256=sha256(inv.ROOT/'selection.lock.json'),inputs=['A_Z0','T','P'],C_D_labels_read=False,
       dependencies={n:sha256(core.PROJECT/'tools'/n) for n in ['sequence_innovation_spike.py','explore_sequence_mechanisms.py','sequence_grid_spike.py','sequence_mechanism_spike.py']}))
    for seed in (42,17):
        ds={n:prepare(a,n,seed) for n in ('train','inner')};d=ds['train'];fit=a['split_fit'];inner=a['split_inner']
        counts=d['mask'][:,:,fit].sum((0,1)).clamp_min(1)
        for arm in ('self','graph'):
            torch.manual_seed(seed);model=GraphCorrection().cuda();rng=np.random.default_rng(seed)
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            best,p=evaluate(model,ds['inner'],g,arm,inner);best=best['macro_rmse'];state=copy.deepcopy(model.state_dict());selected=0
            assert np.max(np.abs(p-ds['inner']['base'][:,:,inner].cpu().numpy()))==0
            trace=[]
            for step in range(1,601):
                row=[int(rng.integers(len(d['x'])))];loc=rng.choice(len(fit),32,replace=False);ids=fit[loc]
                p=predict(model,d,g,arm,row,ids);w=d['mask'][row][:,:,ids]/counts[loc]
                loss=(((p-d['y'][row][:,:,ids])/10)**2*w).sum()/w.sum().clamp_min(1e-8)
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
                if step%100==0:
                    score,_=evaluate(model,ds['inner'],g,arm,inner);trace.append(dict(step=step,inner=score))
                    if score['macro_rmse']<best:best=score['macro_rmse'];state=copy.deepcopy(model.state_dict());selected=step
                    print(json.dumps(dict(seed=seed,arm=arm,step=step,best=best,selected=selected)),flush=True)
            path=ROOT/f'{arm}_{seed}.pt';torch.save(dict(model=state,selected_step=selected),path)
            history[f'{arm}_{seed}']=trace;locked[f'{arm}_{seed}']=dict(sha256=sha256(path),selected_step=selected,inner_macro_rmse=best)
        del ds;torch.cuda.empty_cache()
    write_json(ROOT/'selection.lock.json',locked)
    hours=np.array([int(np.datetime64(f'2018-{m:02d}-27','h').astype('i8'))+np.arange(-6,24) for m in range(1,13)])
    store=core.PackedStore(core.DATA,purpose='train');a['reserved_hours']=hours
    a['reserved_raw']=store.context_values(hours.ravel())[...,:3].reshape(12,30,len(a['aids']),3)
    a['reserved_y']=np.full((12,30,len(a['heights'])),np.nan,dtype='f4')
    a['reserved_y'][...,a['bids']]=store.targets('B',hours.ravel()).reshape(12,30,len(a['bids']))
    results={}
    for seed in (42,17):
        for name in ('confirmation','reserved'):
            d=prepare(a,name,seed);ids=a['split_outer']
            old=np.load(inv.ROOT/f'history_{seed}_{name}.npz')
            assert np.array_equal(old['mask'],d['mask'][:,:,ids].cpu().numpy())
            assert np.array_equal(old['station_ids'],ids)
            assert np.max(np.abs(old['prediction']-d['base'][:,:,ids].cpu().numpy()))<1e-4
            for arm in ('self','graph'):
                model=GraphCorrection().cuda();model.load_state_dict(torch.load(ROOT/f'{arm}_{seed}.pt',weights_only=False)['model'])
                score,p=evaluate(model,d,g,arm,ids);delta=p-old['prediction'];s=old['samples']+delta[:,None]
                results[f'{arm}_{seed}_{name}']=dict(point=score,k64=inv.probability_score(s,old['target'],old['mask']))
                np.savez_compressed(ROOT/f'{arm}_{seed}_{name}.npz',prediction=p,samples=s,target=old['target'],mask=old['mask'],station_ids=ids,hours=old['hours'])
            del d
    write_json(ROOT/'results.json',dict(results=results,history=history,selection=locked,runtime=dict(seconds=time.monotonic()-start,
       device=torch.cuda.get_device_name(),code_sha256=sha256(__file__),selection_sha256=sha256(ROOT/'selection.lock.json'))))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
