"""Preflight and post-run checks for differentiable observation-space diffusion."""
import json
import sys
import numpy as np
import torch
import diffusion_observation_train as e
from gnss_field.artifacts import write_json,sha256


def main():
    torch.set_num_threads(2);a=dict(np.load(e.d.old.exp.ROOT/'data.npz'));post='--post' in sys.argv
    if post:
        geometry=np.load(e.ROOT/'geometry.npz');anchors=geometry['anchors'];grid=geometry['grid_basis'];q=geometry['station_basis']
    else:anchors,grid,q=e.geometry(a)
    old=np.load(e.d.ROOT/'geometry.npz');assert np.array_equal(grid[:,:24],old['grid_basis']) and np.array_equal(q[:,:24],old['station_basis'])
    ids=a['split_fit'];gram=q[ids].T@q[ids];orth=float(np.max(np.abs(gram-np.eye(64))));assert orth<1e-4
    back=(grid[e.d.old.QI]*e.d.old.QW[...,None]).sum(1)*np.exp(-a['heights']/3000)[:,None]
    parity=float(np.max(np.abs(back-q)));assert parity<1e-6
    c=torch.tensor(e.d.condition(a,'inner',anchors)[:1,:4],device='cuda');checks=dict(nested_basis=True,orthogonality_error=orth,basis_query_error=parity,models={})
    for rank in (24,64):
        torch.manual_seed(42);model=e.Model(rank).cuda()
        if post:model.load_state_dict(torch.load(e.ROOT/f'r{rank}_temporal_42.pt',weights_only=False)['model'])
        noise=torch.randn(1,4,4,rank,device='cuda');z,_=e.sample(model,c,noise)
        # Real sampler graph, not a surrogate denoiser-only backward pass.
        phi=torch.tensor(q[a['split_fit'][:32],:rank].copy(),device='cuda')
        s=torch.einsum('bktl,nl->bktn',z,phi)
        target=torch.ones(1,4,32,device='cuda');mask=torch.ones_like(target,dtype=torch.bool)
        loss=e.d.old.core.fair_crps(s,target,mask)+.5*e.d.old.core.fair_crps(s[:,:,1:]-s[:,:,:-1],target[:,1:]-target[:,:-1],mask[:,1:]&mask[:,:-1])
        loss.backward();grad=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        assert grad>0 and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        with torch.no_grad():
            cp=c.clone();cp[:,2:]+=100;zp,_=e.sample(model,cp,noise);err=float((zp[:,:,:2]-z.detach()[:,:,:2]).abs().max());assert err==0
        checks['models'][rank]=dict(sampler_gradient_l1=grad,future_perturbation_error=err)
    if post:
        lock=json.loads((e.ROOT/'experiment.lock.json').read_text());r=json.loads((e.ROOT/'results.json').read_text());parent=json.loads((e.d.ROOT/'experiment.lock.json').read_text())
        assert sha256(e.__file__)==lock['code_sha256']==r['runtime']['code_sha256']
        assert sha256(e.ROOT/'PROTOCOL.md')==lock['protocol_sha256']
        assert sha256(e.d.__file__)==lock['diffusion_source_sha256']
        assert sha256(e.d.ROOT/'experiment.lock.json')==lock['parent_lock_sha256']
        assert sha256(e.d.old.exp.ROOT/'data.npz')==lock['data_sha256']
        assert sha256(e.ROOT/'geometry.npz')==lock['geometry_sha256']
        assert sha256(e.ROOT/'selection.lock.json')==r['runtime']['selection_sha256']
        assert sha256(e.d.ROOT/'geometry.npz')==parent['geometry_sha256']
        assert sha256(e.d.old.inv.ROOT/'selection.lock.json')==parent['v04_selection_sha256']
        for n,h in parent['dependencies'].items():assert sha256(e.d.old.core.PROJECT/'tools'/n)==h
        for n,h in parent['source_checkpoints'].items():assert sha256(e.d.old.exp.ROOT/n)==h
        for n,h in r['runtime']['checkpoints'].items():assert sha256(e.ROOT/n)==h
        maxerr=0.;shuffle={}
        for key,score in r['results'].items():
            p=np.load(e.ROOT/f'{key}.npz');rank,arm,seed,name=key.split('_');ref=np.load(e.d.old.inv.ROOT/f'history_{seed}_{name}.npz')
            for f in ('target','mask','hours','station_ids'):assert np.array_equal(p[f],ref[f])
            assert np.isfinite(p['samples']).all()
            prob=e.d.old.inv.probability_score(p['samples'],p['target'],p['mask']);point=e.d.old.inv.point_score(p['prediction'],p['target'],p['mask'])
            for f,v in prob.items():maxerr=max(maxerr,abs(v-score['k64'][f]))
            for f,v in point.items():maxerr=max(maxerr,abs(v-score['k64']['point'][f]))
            if name=='reserved':
                s=p['samples'].copy();rng=np.random.default_rng(771)
                for b in range(len(s)):
                    for t in range(s.shape[2]):s[b,:,t]=s[b,rng.permutation(s.shape[1]),t]
                sc=e.d.old.inv.probability_score(s,p['target'],p['mask']);assert abs(sc['crps']-prob['crps'])<1e-5
                shuffle[key]=dict(original=prob['increment_crps'],shuffled=sc['increment_crps'])
        assert maxerr<1e-5;checks.update(source_hashes=True,same_population=True,saved_metric_max_error=maxerr)
        write_json(e.ROOT/'shuffle.json',shuffle)
        from verify_diffusion_candidates import background_grid
        p=np.load(e.ROOT/'r64_temporal_42_reserved.npz');latent=p['latent'][:1,:8]
        residual=np.einsum('bktl,nl->bktn',latent,grid);background=background_grid(a);field=background[:,None]+residual
        back=(field[:,:,:,e.d.old.QI[p['station_ids']]]*e.d.old.QW[p['station_ids']]).sum(-1)*np.exp(-a['heights'][p['station_ids']]/3000)
        err=float(np.max(np.abs(back-p['samples'][:1,:8])));assert err<1e-3 and np.isfinite(field).all()
        checks['full_grid_roundtrip_mm']=err
        np.savez_compressed(e.ROOT/'example_full_sequence.npz',z0=field.reshape(8,24,144,96),background_z0=background.reshape(24,144,96),hours=p['hours'][0])
    write_json(e.ROOT/('verification.json' if post else 'preflight.json'),dict(checks=checks,code_sha256=sha256(__file__)))
    print(json.dumps(checks))


if __name__=='__main__':main()
