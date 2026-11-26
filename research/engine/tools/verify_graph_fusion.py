"""Independent invariants and K128 checks for the graph spike."""
import json
import numpy as np
import torch
import graph_fusion_spike as gf
from gnss_field.artifacts import write_json,sha256


@torch.no_grad()
def main():
    torch.set_num_threads(2);a=dict(np.load(gf.exp.ROOT/'data.npz'));g=gf.geometry(a)
    results=json.loads((gf.ROOT/'results.json').read_text());lock=json.loads((gf.ROOT/'experiment.lock.json').read_text())
    assert sha256(gf.__file__)==lock['code_sha256']==results['runtime']['code_sha256']
    assert sha256(gf.ROOT/'PROTOCOL.md')==lock['protocol_sha256']
    assert sha256(gf.exp.ROOT/'data.npz')==lock['data_sha256']
    assert sha256(gf.inv.ROOT/'selection.lock.json')==lock['v04_selection_sha256']
    for name,digest in lock['source_checkpoints'].items():assert sha256(gf.exp.ROOT/name)==digest
    assert sha256(gf.ROOT/'selection.lock.json')==results['runtime']['selection_sha256']
    for name,digest in lock['dependencies'].items():assert sha256(gf.core.PROJECT/'tools'/name)==digest
    for name,entry in results['selection'].items():assert sha256(gf.ROOT/f'{name}.pt')==entry['sha256']
    d=gf.prepare(a,'inner',42);model=gf.GraphCorrection().cuda()
    model.load_state_dict(torch.load(gf.ROOT/'graph_42.pt',weights_only=False)['model'])
    ids=a['split_inner'][:20];full=gf.predict(model,d,g,'graph',[0],ids)
    chunks=torch.cat([gf.predict(model,d,g,'graph',[0],part) for part in np.array_split(ids,3)],-1)
    chunk_error=float((full-chunks).abs().max());assert chunk_error<1e-4
    torch.manual_seed(919);perm=torch.randperm(len(a['aids']),device='cuda');inverse=torch.argsort(perm)
    gp=dict(g);gp['neighbors']=inverse[g['neighbors'][perm]];gp['edges']=g['edges'][perm];gp['qn']=inverse[g['qn']]
    dp=dict(d);dp['x']=d['x'][:,:,perm];dp['valid']=d['valid'][:,:,perm]
    perm_error=float((full-gf.predict(model,dp,gp,'graph',[0],ids)).abs().max());assert perm_error<1e-4
    dp=dict(d);dp['x']=d['x'].clone();dp['x'][:,12:]+=10
    causal_error=float((full[:,:12]-gf.predict(model,dp,g,'graph',[0],ids)[:,:12]).abs().max());assert causal_error==0
    empty=dict(d);empty['valid']=torch.zeros_like(d['valid'])
    assert torch.isfinite(gf.predict(model,empty,g,'graph',[0],ids)).all()
    # Verify raw-to-node temporal offsets independently of the network.
    raw=a['inner_raw'].copy();mu=np.zeros((12,30,len(a['heights'])))
    x,_=gf.node_features(a,'inner',mu);ac=dict(a);ac['inner_raw']=raw.copy();ac['inner_raw'][:,18:]+=100
    xc,_=gf.node_features(ac,'inner',mu);assert np.array_equal(x[:,:12],xc[:,:12])
    checks=dict(query_chunk_max_mm=chunk_error,node_permutation_max_mm=perm_error,future_perturbation_max_mm=causal_error,
                raw_feature_causality=True,all_missing_finite=True,source_hashes=True)
    del d,dp
    hours=np.array([int(np.datetime64(f'2018-{m:02d}-27','h').astype('i8'))+np.arange(-6,24) for m in range(1,13)])
    store=gf.core.PackedStore(gf.core.DATA,purpose='train');a['reserved_hours']=hours
    a['reserved_raw']=store.context_values(hours.ravel())[...,:3].reshape(12,30,len(a['aids']),3)
    a['reserved_y']=np.full((12,30,len(a['heights'])),np.nan,dtype='f4')
    a['reserved_y'][...,a['bids']]=store.targets('B',hours.ravel()).reshape(12,30,len(a['bids']))
    scores={};identities=[]
    for seed in (42,17):
        mean=gf.GridMean(27,False).cuda();mean.load_state_dict(torch.load(gf.exp.ROOT/f'distance_{seed}_2000.pt',weights_only=False)['model'])
        gen=gf.GridRandom(True).cuda();gen.load_state_dict(torch.load(gf.exp.ROOT/f'single_{seed}_1400.pt',weights_only=False)['model'])
        for name in ('confirmation','reserved'):
            d=gf.exp.grid_features(a,name,False);res,h=mean(d['x']);d['mu']=d['base']+res;d['h']=h
            _,p=gf.exp.evaluate(gen,d,a['split_outer'],noise_seed=11703,members=128)
            base=(d['mu'][:,6:,a['split_outer']]*d['height'][a['split_outer']]).cpu().numpy()
            old=np.load(gf.inv.ROOT/f'history_{seed}_{name}.npz')
            baseline=p['samples']+(old['prediction']-base)[:,None]
            scores[f'history_{seed}_{name}']=gf.inv.probability_score(baseline,p['target'],p['mask'])
            for arm in ('self','graph'):
                pred=np.load(gf.ROOT/f'{arm}_{seed}_{name}.npz')
                assert np.array_equal(pred['mask'],old['mask']) and np.array_equal(pred['hours'],old['hours'])
                assert np.array_equal(pred['target'],old['target']) and np.array_equal(pred['station_ids'],old['station_ids'])
                assert np.isfinite(pred['prediction']).all() and np.isfinite(pred['samples']).all()
                diff=pred['prediction']-old['prediction']
                identities.append(float(np.max(np.abs((pred['samples']-old['samples'])-diff[:,None]))))
                scores[f'{arm}_{seed}_{name}']=gf.inv.probability_score(baseline+diff[:,None],p['target'],p['mask'])
            del d
    checks['same_population']=True;checks['sample_shift_max_mm']=max(identities);assert max(identities)<1e-9
    write_json(gf.ROOT/'verification.json',dict(checks=checks,k128=scores,code_sha256=sha256(__file__)))
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
