"""Diffusion algebra, gradient, causal conditioning and grid checks."""
import json
import sys
import numpy as np
import torch
from scipy.spatial.distance import cdist
import diffusion_candidates as d
from gnss_field.artifacts import write_json,sha256


@torch.no_grad()
def background_grid(a):
    d.masked.add_reserved(a);raw=a['reserved_raw'][:1];hours=a['reserved_hours'][:1]
    x,base,_=d.old.core.encode_inputs(a,raw,hours,d.old.XY,d.old.TERRAIN)
    model=d.old.GridMean(27,False).cuda()
    model.load_state_dict(torch.load(d.old.exp.ROOT/'distance_42_2000.pt',weights_only=False)['model'])
    mu=base+d.old.core.MeanModel.forward(model,torch.tensor(x,device='cuda'))[0].cpu().numpy()
    amu=(mu[:,:,d.old.QI[a['aids']]]*d.old.QW[a['aids']]).sum(-1)
    valid=np.isfinite(raw[...,0]);err=np.where(valid,raw[...,0]-amu,0)
    dist=cdist(d.old.XY,a['xy'][a['aids']])/1000
    dz=d.old.TERRAIN[:,0,None]*3000-a['heights'][a['aids']][None]
    fields=[]
    for radius in (50,150,300):
        for height in (False,True):
            w=np.exp(-.5*(dist/radius)**2)*(dist<=3*radius)
            if height:w*=np.exp(-.5*(dz/750)**2)
            den=np.einsum('qa,bta->btq',w,valid);num=np.einsum('qa,bta->btq',w,err)
            fields.append(np.divide(num,den,out=np.zeros_like(num),where=den>1e-8))
    feat=np.stack(fields,-1);feat=np.concatenate([feat[:,6-lag:30-lag] for lag in (0,1,3)],-1)
    sel=json.loads((d.old.inv.ROOT/'selection.lock.json').read_text())['42']['history']
    correction=sel['beta'][0]+(feat/np.array(sel['scales']))@np.array(sel['beta'][1:])
    return mu[:,6:]+correction


def main():
    torch.set_num_threads(2);torch.manual_seed(42);a=dict(np.load(d.old.exp.ROOT/'data.npz'))
    anchors,grid,query,_=d.spatial_basis(a);ab=d.schedule()
    x0=torch.randn(128,24,device='cuda');noise=torch.randn_like(x0);k=torch.randint(1,101,(128,),device='cuda');alpha=ab[k,None]
    x=alpha.sqrt()*x0+(1-alpha).sqrt()*noise;v=alpha.sqrt()*noise-(1-alpha).sqrt()*x0
    restored=alpha.sqrt()*x-(1-alpha).sqrt()*v
    error=float((restored-x0).abs().max());assert error<2e-6 and float(ab[0])==1.
    c=torch.tensor(d.condition(a,'inner',anchors)[:1,:3],device='cuda')
    checks=dict(v_identity_error=error,alpha_clean=float(ab[0]),alpha_terminal=float(ab[-1]),models={})
    post='--post' in sys.argv
    for kind in d.KINDS:
        model=d.Model(kind).cuda()
        if post:model.load_state_dict(torch.load(d.ROOT/f'{kind}_42.pt',weights_only=False)['model'])
        out,aux=model(x,torch.zeros_like(x),c[:,0].expand(128,-1),k.float(),torch.zeros(128,device='cuda'))
        loss=(out*out).mean()+.01*aux;loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        first,_=d.sample(model,c,4,771);changed=c.clone();changed[:,2]+=100
        second,_=d.sample(model,changed,4,771)
        causal=float((first[:,:,:2]-second[:,:,:2]).abs().max());assert causal==0 and torch.isfinite(first).all()
        checks['models'][kind]=dict(causal_latent_error=causal,finite_gradient=True)
    # Condition construction independent of all B target values.
    clone=dict(a);clone['inner_y']=np.full_like(a['inner_y'],999999)
    assert np.array_equal(d.condition(a,'inner',anchors),d.condition(clone,'inner',anchors))
    qc=(grid[d.old.QI]*d.old.QW[...,None]).sum(1)*np.exp(-a['heights']/3000)[:,None]
    err=float(np.max(np.abs(query-qc)));assert err<1e-6;checks['basis_query_error']=err
    if post:
        lock=json.loads((d.ROOT/'experiment.lock.json').read_text());results=json.loads((d.ROOT/'results.json').read_text())
        assert sha256(d.__file__)==lock['code_sha256']==results['runtime']['code_sha256']
        assert sha256(d.ROOT/'ARCHITECTURE.md')==lock['architecture_sha256']
        assert sha256(d.old.exp.ROOT/'data.npz')==lock['data_sha256']
        assert sha256(d.ROOT/'geometry.npz')==lock['geometry_sha256']
        assert sha256(d.old.inv.ROOT/'selection.lock.json')==lock['v04_selection_sha256']
        assert sha256(d.ROOT/'selection.lock.json')==results['runtime']['selection_sha256']
        for name,h in lock['source_checkpoints'].items():assert sha256(d.old.exp.ROOT/name)==h
        for name,h in lock['dependencies'].items():assert sha256(d.old.core.PROJECT/'tools'/name)==h
        for name,h in results['runtime']['checkpoints'].items():assert sha256(d.ROOT/name)==h
        max_error=0.
        for key,score in results['results'].items():
            pred=np.load(d.ROOT/f'{key}.npz');parts=key.split('_');ref=np.load(d.old.inv.ROOT/f'history_{parts[1]}_{parts[2]}.npz')
            for field in ('station_ids','hours','target','mask'):assert np.array_equal(pred[field],ref[field])
            assert np.isfinite(pred['samples']).all()
            point=d.old.inv.point_score(pred['prediction'],pred['target'],pred['mask'])
            prob=d.old.inv.probability_score(pred['samples'],pred['target'],pred['mask'])
            for field,val in point.items():max_error=max(max_error,abs(val-score['k64']['point'][field]))
            for field,val in prob.items():max_error=max(max_error,abs(val-score['k64'][field]))
        assert max_error<1e-5;checks['saved_metric_max_error']=max_error;checks['same_population']=True;checks['source_hashes']=True
        pred=np.load(d.ROOT/'transformer_42_reserved.npz');z=pred['latent'][:1,:8]
        field=np.einsum('bktl,nl->bktn',z,grid)
        back=(field[:,:,:,d.old.QI[pred['station_ids']]]*d.old.QW[pred['station_ids']]).sum(-1)*np.exp(-a['heights'][pred['station_ids']]/3000)
        direct=np.einsum('bktl,nl->bktn',z,query[pred['station_ids']]);err=float(np.max(np.abs(back-direct)));assert err<1e-4
        checks['residual_grid_roundtrip_mm']=err
        np.savez_compressed(d.ROOT/'example_residual_sequence.npz',residual_z0=field.reshape(8,24,144,96),hours=pred['hours'][0])
        background=background_grid(a);full=background[:,None]+field
        back=(full[:,:,:,d.old.QI[pred['station_ids']]]*d.old.QW[pred['station_ids']]).sum(-1)*np.exp(-a['heights'][pred['station_ids']]/3000)
        error=float(np.max(np.abs(back-pred['samples'][:1,:8])));assert error<1e-3
        checks['full_grid_roundtrip_mm']=error
        np.savez_compressed(d.ROOT/'example_full_sequence.npz',z0=full.reshape(8,24,144,96),background_z0=background.reshape(24,144,96),hours=pred['hours'][0])
    write_json(d.ROOT/('verification.json' if post else 'preflight.json'),dict(checks=checks,code_sha256=sha256(__file__)))
    print(json.dumps(checks))


if __name__=='__main__':main()
