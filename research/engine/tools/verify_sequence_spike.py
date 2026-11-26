"""Independent arithmetic, causality and published-grid checks for the spike."""
import json
import numpy as np
import torch
from scipy.ndimage import uniform_filter
from sequence_mechanism_spike import (ROOT, DATA, PackedStore, query_table, encode_inputs,
    MeanModel, RandomField, fair_crps, metrics)
from gnss_field.artifacts import write_json, sha256


def main():
    torch.set_num_threads(2)
    a=dict(np.load(ROOT/'data.npz')); store=PackedStore(DATA,purpose='context')
    receipt=json.loads((ROOT/'data_receipt.json').read_text())
    assert sha256(ROOT/'data.npz')==receipt['data_sha256']
    checks={}
    # Fair CRPS must agree with direct ordered-pair expression.
    torch.manual_seed(8)
    s=torch.randn(2,8,4,5); y=torch.randn(2,4,5); m=torch.rand(2,4,5)>.2
    direct=((s-y[:,None]).abs().mean(1)-(s[:,:,None]-s[:,None,:]).abs().sum((1,2))/(2*8*7))
    assert torch.allclose(fair_crps(s,y,m),(direct*m).sum()/m.sum(),atol=1e-6)
    checks['fair_crps_ordered_pair_parity']=True
    model=MeanModel(26,True)
    x=torch.randn(1,30,5,26); changed=x.clone(); changed[:,15:]+=10
    with torch.no_grad():
        # Check latent encoder too: zero output head alone would make this trivial.
        assert torch.equal(model(x)[1][:,:15],model(changed)[1][:,:15])
    checks['causal_encoder_future_perturbation']=True
    g=RandomField(True); noise=torch.randn(1,16,30,8); h=torch.randn(1,30,5,32)
    perm=torch.tensor([4,1,3,2,0])
    assert torch.allclose(g(h[:,:,perm],noise),g(h,noise)[:,:,:,perm],atol=1e-6)
    checks['spatial_query_permutation']=True
    # Common spatial draws preserve per-query results when chunks change.
    assert torch.allclose(torch.cat([g(h[:,:,:2],noise),g(h[:,:,2:],noise)],-1),g(h,noise),atol=1e-6)
    checks['spatial_chunk_consistency']=True
    # Distinct train/inner/outer intervals including warmup.
    for p,q in [('train','inner'),('train','outer'),('inner','outer')]:
        assert not np.intersect1d(a[p+'_hours'],a[q+'_hours']).size
    checks['time_partition_disjoint']=True
    for p,q in [('fit','inner'),('fit','outer'),('inner','outer')]:
        assert not np.intersect1d(a['split_'+p],a['split_'+q]).size
    checks['station_partition_disjoint']=True
    # Fixed source checkpoint integrity.
    r=json.loads((ROOT/'results.json').read_text())
    for name,digest in r['runtime']['checkpoints'].items():assert sha256(ROOT/name)==digest
    checks['checkpoint_hashes']=True
    bootstrap={}
    for seed in ('42','17'):
        bootstrap[seed]={}
        for metric in ('crps','increment_crps'):
            delta=np.array([v[metric] for v in r[seed]['ar']['per_sequence']])-np.array([v[metric] for v in r[seed]['independent']['per_sequence']])
            rng=np.random.default_rng(20261123)
            boot=delta[rng.integers(0,12,(5000,12))].mean(1)
            bootstrap[seed][metric]=dict(equal_sequence_mean_difference=float(delta.mean()),
                ci95=np.quantile(boot,[.025,.975]).tolist(),unit='12 monthly 24h blocks, exploratory')
    # Query the actual publication grid, not the station's direct coordinate decoder.
    ids=a['split_outer']; _,qi,qw,_=query_table(store)
    nodes,inverse=np.unique(qi[ids],return_inverse=True); inverse=inverse.reshape(-1,4)
    grid=store.grid; z=grid['height']; gy,gx=np.gradient(z,25000.,25000.)
    rough=np.sqrt(np.maximum(uniform_filter(z*z,5)-uniform_filter(z,5)**2,0))
    terrain=np.stack([z/3000,gx/.1,gy/.1,rough/500],-1).reshape(-1,4)
    xy=np.c_[grid['x'].ravel(),grid['y'].ravel()]
    # .reshape above is correct for stack(...,-1): preserve per-point feature order.
    x,base,_=encode_inputs(a,a['outer_raw'],a['outer_hours'],xy[nodes],terrain[nodes])
    xt=torch.tensor(x,device='cuda'); bt=torch.tensor(base,device='cuda')
    _,_,support=encode_inputs(a,a['outer_raw'],a['outer_hours'],a['xy'][ids],a['terrain'][ids])
    target=a['outer_y'][:,:,ids]; mask=np.isfinite(target)&support&a['qvalid'][ids]
    yt=torch.tensor(np.nan_to_num(target[:,6:]),device='cuda'); mt=torch.tensor(mask[:,6:],device='cuda')
    weights=torch.tensor(qw[ids],dtype=torch.float32,device='cuda'); factor=torch.tensor(np.exp(-a['heights'][ids]/3000),dtype=torch.float32,device='cuda')
    canonical={}
    with torch.no_grad():
        for seed in (42,17):
            mean=MeanModel(x.shape[-1],True).cuda(); mean.load_state_dict(torch.load(ROOT/f'gru_{seed}.pt',weights_only=True))
            residual,h=mean(xt); mu=bt+residual
            muq=(mu[:,:,inverse]*weights).sum(-1)*factor
            canonical[str(seed)]={'mean':metrics(muq[:,6:].cpu().numpy().reshape(-1,len(ids)),yt.cpu().numpy().reshape(-1,len(ids)),mt.cpu().numpy().reshape(-1,len(ids)))}
            for correlated in (False,True):
                name='ar' if correlated else 'independent'; gen=RandomField(correlated).cuda()
                gen.load_state_dict(torch.load(ROOT/f'{name}_{seed}.pt',weights_only=True))
                torch.manual_seed(9301); noise=torch.randn(len(x),64,30,8,device='cuda')
                fields=mu[:,None]+gen(h,noise)
                samples=((fields[:,:,:,inverse]*weights).sum(-1)*factor)[:,:,6:]
                low,high=torch.quantile(samples,torch.tensor([.05,.95],device='cuda'),dim=1)
                canonical[str(seed)][name]=dict(crps=float(fair_crps(samples,yt,mt)),
                    increment_crps=float(fair_crps(samples[:,:,1:]-samples[:,:,:-1],yt[:,1:]-yt[:,:-1],mt[:,1:]&mt[:,:-1])),
                    coverage90=float((((yt>=low)&(yt<=high))&mt).sum()/mt.sum()))
                np.savez_compressed(ROOT/f'grid_query_{name}_{seed}.npz',samples=samples.cpu().numpy(),target=yt.cpu().numpy(),mask=mt.cpu().numpy())
        # Full-grid sequence, fixed first outer day, last trained seed/AR arm.
        torch.manual_seed(9301); noise=torch.randn(1,8,30,8,device='cuda'); chunks=[]
        for start in range(0,len(xy),512):
            stop=start+512
            fx,fb,_=encode_inputs(a,a['outer_raw'][:1],a['outer_hours'][:1],xy[start:stop],terrain[start:stop])
            residual,h=mean(torch.tensor(fx,device='cuda'))
            fields=torch.tensor(fb,device='cuda')[:,None]+residual[:,None]+gen(h,noise)
            chunks.append(fields[0,:,6:].cpu().numpy())
        field=np.concatenate(chunks,-1).reshape(8,24,*z.shape)
        assert np.isfinite(field).all()
        np.savez_compressed(ROOT/'example_grid_sequence.npz',Z0_members=field,land=grid['region'],hours=a['outer_hours'][0,6:])
    checks['full_grid_sequence_finite']=list(field.shape)
    write_json(ROOT/'verification.json',dict(checks=checks,paired_bootstrap=bootstrap,canonical_grid_query=canonical,
        distinction='Training scores direct coordinate queries; publication-grid query is a post-training robustness check. Future formal training must use one locked operator.',
        verifier_sha256=sha256(__file__)))
    print(json.dumps(dict(checks=checks,paired_bootstrap=bootstrap,canonical_grid_query=canonical)),flush=True)


if __name__=='__main__':main()
