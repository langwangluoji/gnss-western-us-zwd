"""Budget and mechanism contrasts, with a separately reserved calendar check."""
import copy
import json
import time
import numpy as np
import torch
from scipy.spatial.distance import cdist
import sequence_mechanism_spike as core
from sequence_grid_spike import GridMean, GridRandom, XY, TERRAIN, QI, QW
from gnss_field.artifacts import write_json, sha256

ROOT=core.PROJECT/'results/20261123_sequence_mechanisms_v03'
SOURCE=core.PROJECT/'results/20261123_sequence_spike_v02_grid'


def load_data():
    receipt=json.loads((SOURCE/'data_receipt.json').read_text())
    assert sha256(SOURCE/'data.npz')==receipt['data_sha256']
    a=dict(np.load(SOURCE/'data.npz'))
    hours=np.array([int(np.datetime64(f'2018-{m:02d}-22','h').astype('i8'))+np.arange(-6,24) for m in range(1,13)])
    store=core.PackedStore(core.DATA,purpose='train')
    a['confirmation_hours']=hours
    a['confirmation_raw']=store.context_values(hours.ravel())[...,:3].reshape(12,30,len(a['aids']),3)
    y=np.full((12,30,len(a['heights'])),np.nan,dtype='f4')
    y[...,a['bids']]=store.targets('B',hours.ravel()).reshape(12,30,len(a['bids']))
    a['confirmation_y']=y
    for name in ('train','inner','outer'):
        assert not np.intersect1d(hours,a[name+'_hours']).size
    write_json(ROOT/'experiment.lock.json',dict(source_data_sha256=receipt['data_sha256'],
        code_sha256=sha256(__file__),protocol_sha256=sha256(ROOT/'PROTOCOL.md'),
        dependencies={name:sha256(core.PROJECT/'tools'/name) for name in ['sequence_mechanism_spike.py','sequence_grid_spike.py']},
        confirmation_hours=hours.tolist(),confirmation_not_for_selection=True,mean_budgets=[500,2000],
        random_budgets=[350,1400],seeds=[42,17],inputs=['A_Z0','T','P'],C_D_labels_read=False,
        historical_cohort_PWV_selection=True))
    np.savez(ROOT/'data.npz',**a)
    return a


def grid_features(a,name,height_aware):
    raw,hours=a[name+'_raw'],a[name+'_hours']
    nodes,reverse=np.unique(QI,return_inverse=True); index=reverse.reshape(-1,4)
    x,base,_=core.encode_inputs(a,raw,hours,XY[nodes],TERRAIN[nodes])
    if height_aware:
        dist=cdist(XY[nodes],a['xy'][a['aids']])/1000
        dz=TERRAIN[nodes,0,None]*3000-a['heights'][a['aids']][None]
        valid=np.isfinite(raw).all(-1); normalized=np.nan_to_num((raw-a['mean'])/a['std'])
        features=[]
        for radius in (50,150,300):
            w=np.exp(-.5*(dist/radius)**2-.5*(dz/750)**2)*(dist<=3*radius)
            den=np.einsum('qa,bta->btq',w,valid)+1e-8
            avg=np.einsum('qa,btav->btqv',w,normalized*valid[...,None])/den[...,None]
            second=np.einsum('qa,bta->btq',w,normalized[...,0]**2*valid)/den
            features.extend([avg,np.sqrt(np.maximum(second-avg[...,0]**2,0))[...,None],np.log1p(den)[...,None]])
        x[...,:15]=np.concatenate(features,-1)
    _,_,support=core.encode_inputs(a,raw,hours,a['xy'],a['terrain'])
    x=x[:,:,index];weights=np.broadcast_to(QW,x.shape[:-1])[...,None].astype('f4')
    x=np.concatenate([x,weights],-1)
    base=(base[:,:,index]*QW).sum(-1).astype('f4')
    y=a[name+'_y'];mask=np.isfinite(y)&support&a['qvalid']
    return {k:torch.as_tensor(v,device='cuda') for k,v in dict(x=x,base=base,y=np.nan_to_num(y),mask=mask,
                  height=np.exp(-a['heights']/3000).astype('f4')).items()}


def fit_mean(ds,a,seed,arm):
    torch.manual_seed(seed);model=GridMean(27,False).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001);rng=np.random.default_rng(seed)
    ids=a['split_fit'];iid=a['split_inner'];best=float('inf');best_step=0;history=[];states={}
    d=ds['train'];start=time.monotonic()
    for step in range(1,2001):
        rows=rng.integers(len(d['x']),size=4)
        p=(model(d['x'][rows][:,:,ids])[0]+d['base'][rows][:,:,ids])*d['height'][ids]
        m=d['mask'][rows][:,6:,ids];y=d['y'][rows][:,6:,ids]
        loss=(((p[:,6:]-y)/10)**2*m).sum()/m.sum()
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        if step%100==0:
            inner=core.mean_eval(model,ds['inner'],iid)
            train=core.mean_eval(model,{k:v[:4] if k!='height' else v for k,v in d.items()},ids)
            history.append(dict(step=step,inner=inner,train_fixed4=train))
            if inner['macro_rmse']<best:
                best=inner['macro_rmse'];state=copy.deepcopy(model.state_dict());best_step=step
            if step%500==0:print(json.dumps(dict(stage='mean',seed=seed,arm=arm,step=step,inner=best)),flush=True)
        if step in (500,2000):
            states[step]=copy.deepcopy(state)
            torch.save(dict(model=state,selected_step=best_step,budget=step,seed=seed),ROOT/f'{arm}_{seed}_{step}.pt')
    return states,dict(history=history,seconds=time.monotonic()-start)


@torch.no_grad()
def mean_prediction(state,d,ids):
    model=GridMean(27,False).cuda();model.load_state_dict(state)
    p=(model(d['x'][:,:,ids])[0]+d['base'][:,:,ids])*d['height'][ids]
    y=d['y'][:,6:,ids];m=d['mask'][:,6:,ids];p=p[:,6:]
    return core.metrics(p.cpu().numpy().reshape(-1,len(ids)),y.cpu().numpy().reshape(-1,len(ids)),m.cpu().numpy().reshape(-1,len(ids))),dict(prediction=p.cpu().numpy(),target=y.cpu().numpy(),mask=m.cpu().numpy())


class MultiScaleRandom(GridRandom):
    def __init__(self):
        super().__init__(True)
        self.rho_raw=torch.nn.Parameter(torch.zeros(8))


@torch.no_grad()
def evaluate(gen,d,ids,noise_seed=9301,members=64):
    torch.manual_seed(noise_seed)
    noise=torch.randn(len(d['h']),members,30,8,device='cuda')
    s=((d['mu'][:,:,ids][:,None]+gen(d['h'][:,:,ids],noise))*d['height'][ids])[:,:,6:]
    y=d['y'][:,6:,ids];m=d['mask'][:,6:,ids]
    low,high=torch.quantile(s,torch.tensor([.05,.95],device='cuda'),dim=1)
    point=core.metrics(s.mean(1).cpu().numpy().reshape(-1,len(ids)),y.cpu().numpy().reshape(-1,len(ids)),m.cpu().numpy().reshape(-1,len(ids)))
    point.update(crps=float(core.fair_crps(s,y,m)),increment_crps=float(core.fair_crps(s[:,:,1:]-s[:,:,:-1],y[:,1:]-y[:,:-1],m[:,1:]&m[:,:-1])),
        spatial_pair_crps=float(core.fair_crps(s[:,:,:,1:]-s[:,:,:,:-1],y[:,:,1:]-y[:,:,:-1],m[:,:,1:]&m[:,:,:-1])),
        coverage90=float((((y>=low)&(y<=high))&m).sum()/m.sum()),width90=float(((high-low)*m).sum()/m.sum()),
        rho=(.98*gen.rho_raw.tanh()).cpu().numpy().tolist(),members=members)
    return point,dict(samples=s.cpu().numpy(),target=y.cpu().numpy(),mask=m.cpu().numpy())


def fit_random(ds,a,seed,multi):
    torch.manual_seed(seed);gen=MultiScaleRandom().cuda() if multi else GridRandom(True).cuda()
    opt=torch.optim.AdamW([{'params':gen.head.parameters(),'lr':.002},{'params':[gen.rho_raw],'lr':.02}])
    rng=np.random.default_rng(seed);best=float('inf');states={};history=[];ids=a['split_fit'];start=time.monotonic()
    d=ds['train'];arm='multi' if multi else 'single'
    for step in range(1,1401):
        rows=rng.integers(len(d['h']),size=4);noise=torch.randn(4,8,30,8,device='cuda')
        s=(d['mu'][rows][:,:,ids][:,None]+gen(d['h'][rows][:,:,ids],noise))*d['height'][ids]
        loss=core.objective(s[:,:,6:],d['y'][rows][:,6:,ids],d['mask'][rows][:,6:,ids])
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(gen.parameters(),2);opt.step()
        if step%175==0:
            with torch.random.fork_rng(devices=[0]):score,_=evaluate(gen,ds['inner'],a['split_inner'],members=32)
            criterion=score['crps']+.5*score['increment_crps']
            history.append(dict(step=step,inner=score))
            if criterion<best:best=criterion;state=copy.deepcopy(gen.state_dict());best_step=step
            if step%350==0:print(json.dumps(dict(stage='random',seed=seed,arm=arm,step=step,inner=best)),flush=True)
        if step in (350,1400):
            states[step]=copy.deepcopy(state)
            torch.save(dict(model=state,selected_step=best_step,budget=step,seed=seed),ROOT/f'{arm}_{seed}_{step}.pt')
    return states,dict(history=history,seconds=time.monotonic()-start)


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    start=time.monotonic();a=load_data();saved={};results={};models={}
    # Confirmation features/labels remain on CPU until all training is finished.
    for geometry in ('distance','height'):
        ds={name:grid_features(a,name,geometry=='height') for name in ('train','inner')}
        for seed in (42,17):
            states,history=fit_mean(ds,a,seed,geometry)
            models[(geometry,seed)]=states;results[f'{geometry}_{seed}']=history
        del ds;torch.cuda.empty_cache()
    ds={name:grid_features(a,name,False) for name in ('train','inner')}
    for seed in (42,17):
        model=GridMean(27,False).cuda();model.load_state_dict(models[('distance',seed)][2000]);model.eval()
        with torch.no_grad():
            for d in ds.values():
                chunks=[model(d['x'][i:i+4]) for i in range(0,len(d['x']),4)]
                d['mu']=d['base']+torch.cat([v[0] for v in chunks]);d['h']=torch.cat([v[1] for v in chunks])
        for multi in (False,True):
            arm='multi' if multi else 'single';states,history=fit_random(ds,a,seed,multi)
            saved[(arm,seed)]=states;results[f'{arm}_{seed}']=history
    del ds;torch.cuda.empty_cache()
    # Only now evaluate all locked arms on outer and reserved confirmation weather.
    for geometry in ('distance','height'):
        for population in ('outer','confirmation'):
            d=grid_features(a,population,geometry=='height');ids=a['split_outer']
            for seed in (42,17):
                for budget in (500,2000):
                    score,pred=mean_prediction(models[(geometry,seed)][budget],d,ids)
                    results[f'{geometry}_{seed}'].setdefault(population,{})[str(budget)]=score
                    np.savez_compressed(ROOT/f'{geometry}_{seed}_{budget}_{population}.npz',**pred,station_ids=ids,hours=a[population+'_hours'][:,6:])
                if geometry=='distance':
                    model=GridMean(27,False).cuda();model.load_state_dict(models[('distance',seed)][2000])
                    with torch.no_grad():res,h=model(d['x']);d['mu']=d['base']+res;d['h']=h
                    for arm in ('single','multi'):
                        gen=MultiScaleRandom().cuda() if arm=='multi' else GridRandom(True).cuda()
                        for budget in (350,1400):
                            gen.load_state_dict(saved[(arm,seed)][budget]);score,pred=evaluate(gen,d,ids)
                            check,_=evaluate(gen,d,ids,noise_seed=11703,members=128)
                            results[f'{arm}_{seed}'].setdefault(population,{})[str(budget)]=dict(k64=score,k128_independent_noise=check)
                            np.savez_compressed(ROOT/f'{arm}_{seed}_{budget}_{population}.npz',**pred,station_ids=ids,hours=a[population+'_hours'][:,6:])
            del d;torch.cuda.empty_cache()
    results['runtime']=dict(seconds=time.monotonic()-start,device=torch.cuda.get_device_name(),
         code_sha256=sha256(__file__),data_sha256=sha256(ROOT/'data.npz'),checkpoints={p.name:sha256(p) for p in ROOT.glob('*.pt')})
    write_json(ROOT/'results.json',results)
    print(json.dumps(dict(stage='complete',seconds=results['runtime']['seconds'])),flush=True)


if __name__=='__main__':main()
