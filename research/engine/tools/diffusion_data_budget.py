"""Adaptive paired data/step expansion with fixed latent coordinates."""
import copy
import json
import time
import numpy as np
import torch
import diffusion_candidates as d
from gnss_field.artifacts import write_json,sha256

ROOT=d.old.core.PROJECT/'results/20261123_diffusion_data_v08'


def expand(a):
    extra=np.array([int(np.datetime64(f'{year}-{month:02d}-{day:02d}','h').astype('i8'))+np.arange(-6,24)
                    for year in (2016,2017) for month in range(1,13) for day in (4,12,22)])
    for name in ('train','inner','outer','confirmation'):assert not np.intersect1d(extra,a[name+'_hours']).size
    store=d.old.core.PackedStore(d.old.core.DATA,purpose='train')
    raw=store.context_values(extra.ravel())[...,:3].reshape(72,30,len(a['aids']),3)
    y=np.full((72,30,len(a['heights'])),np.nan,dtype='f4')
    y[...,a['bids']]=store.targets('B',extra.ravel()).reshape(72,30,len(a['bids']))
    extended=dict(a)
    for suffix,values in [('hours',extra),('raw',raw),('y',y)]:extended['train_'+suffix]=np.concatenate([a['train_'+suffix],values])
    return extended,extra


def fit(ds,a,q,seed,arm):
    torch.manual_seed(seed);model=d.Model('transformer').cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    rng=torch.Generator(device='cuda').manual_seed(seed+1000)
    z=ds['train']['z'];prev=torch.cat([torch.zeros_like(z[:,:1]),z[:,:-1]],1).reshape(-1,24)
    first=torch.zeros(z.shape[:2],device='cuda');first[:,0]=1;first=first.flatten()
    z=z.reshape(-1,24);c=ds['train']['c'].reshape(-1,124);ab=d.schedule();best=float('inf');trace=[];start=time.monotonic();match=0.
    prior=json.loads((d.ROOT/'results.json').read_text())['selection'][f'transformer_{seed}']['trace']
    for step in range(1,3001):
        idx=torch.randint(len(z),(128,),device='cuda',generator=rng);target=z[idx]
        k=torch.randint(1,101,(128,),device='cuda',generator=rng);eps=torch.randn(target.shape,device='cuda',generator=rng);alpha=ab[k,None]
        x=alpha.sqrt()*target+(1-alpha).sqrt()*eps;v=alpha.sqrt()*eps-(1-alpha).sqrt()*target
        pred,aux=model(x,prev[idx],c[idx],k.float(),first[idx]);loss=((pred-v)**2).mean()+.01*aux
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        if step%300==0:
            with torch.random.fork_rng(devices=[0]):score,_=d.evaluate(model,ds['inner'],ds['train'],q,a['split_inner'],16)
            criterion=score['crps']+.5*score['increment_crps'];trace.append(dict(step=step,loss=float(loss.detach()),inner=score))
            if criterion<best:best=criterion;state=copy.deepcopy(model.state_dict());selected=step
            if arm=='original' and step<=1500:
                reference=prior[step//300-1]['inner'];delta=max(abs(score[k]-reference[k]) for k in ('crps','increment_crps'))
                match=max(match,delta);assert delta<1e-5,(seed,step,delta)
            if step%600==0:print(json.dumps(dict(seed=seed,arm=arm,step=step,best=best,selected=selected)),flush=True)
    torch.save(dict(model=state,selected_step=selected),ROOT/f'{arm}_{seed}.pt')
    return dict(trace=trace,selected_step=selected,seconds=time.monotonic()-start,original_trajectory_max_error=match if arm=='original' else None)


def main():
    ROOT.mkdir(exist_ok=True);torch.set_num_threads(2);start=time.monotonic()
    a=dict(np.load(d.old.exp.ROOT/'data.npz'));source=json.loads((d.ROOT/'experiment.lock.json').read_text())
    assert sha256(d.old.exp.ROOT/'data.npz')==source['data_sha256'];assert sha256(d.__file__)==source['code_sha256']
    geometry=dict(np.load(d.ROOT/'geometry.npz'));anchors=geometry['anchors'];query=geometry['station_basis'];q=torch.tensor(query,device='cuda')
    extended,extra=expand(a);np.savez(ROOT/'expanded_data.npz',**extended)
    write_json(ROOT/'experiment.lock.json',dict(code_sha256=sha256(__file__),protocol_sha256=sha256(ROOT/'PROTOCOL.md'),
        diffusion_source_sha256=sha256(d.__file__),source_experiment_lock_sha256=sha256(d.ROOT/'experiment.lock.json'),
        source_results_sha256=sha256(d.ROOT/'results.json'),expanded_data_sha256=sha256(ROOT/'expanded_data.npz'),extra_hours=extra.tolist(),C_D_labels_read=False))
    selection={};diagnostics={}
    for seed in (42,17):
        norm=torch.load(d.ROOT/f'normalization_{seed}.pt',weights_only=False)
        inner=d.prepare(a,'inner',seed,anchors,query)
        for arm,data in [('original',a),('expanded',extended)]:
            train=d.prepare(data,'train',seed,anchors,query)
            if arm=='expanded':train['z']=(train['z']*train['zs']+train['zm']-norm['zm'])/norm['zs']
            else:
                assert torch.equal(train['zm'],norm['zm']) and torch.equal(train['zs'],norm['zs'])
            train.update(norm);diagnostics[f'{arm}_{seed}']=train['latent_diagnostics']
            selection[f'{arm}_{seed}']=fit(dict(train=train,inner=inner),a,q,seed,arm)
            del train
    write_json(ROOT/'selection.lock.json',selection);d.masked.add_reserved(a);results={}
    for seed in (42,17):
        norm=torch.load(d.ROOT/f'normalization_{seed}.pt',weights_only=False)
        for name in ('confirmation','reserved'):
            data=d.prepare(a,name,seed,anchors,query);ids=a['split_outer'];ref=np.load(d.old.inv.ROOT/f'history_{seed}_{name}.npz')
            assert np.array_equal(ref['mask'],data['mask'][:,:,ids].cpu().numpy())
            for arm in ('original','expanded'):
                model=d.Model('transformer').cuda();model.load_state_dict(torch.load(ROOT/f'{arm}_{seed}.pt',weights_only=False)['model'])
                score,p=d.evaluate(model,data,norm,q,ids);check,_=d.evaluate(model,data,norm,q,ids,128,11703)
                key=f'{arm}_{seed}_{name}';results[key]=dict(k64=score,k128=check)
                np.savez_compressed(ROOT/f'{key}.npz',**p,station_ids=ids,hours=ref['hours'])
                print(json.dumps(dict(stage='eval',key=key,k128=check)),flush=True)
    write_json(ROOT/'results.json',dict(results=results,selection=selection,latent_diagnostics=diagnostics,runtime=dict(seconds=time.monotonic()-start,code_sha256=sha256(__file__),
        selection_sha256=sha256(ROOT/'selection.lock.json'),checkpoints={p.name:sha256(p) for p in ROOT.glob('*.pt')})))


if __name__=='__main__':main()
