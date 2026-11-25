import importlib.util
from pathlib import Path
import torch

path=Path(__file__).with_name('native_process_models.py')
spec=importlib.util.spec_from_file_location('native_process_models',path)
module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_process_models as module
from . import native_process_ops as ops


def inputs():
    torch.manual_seed(64);c=torch.randn(2,5,175);eta=torch.randn(2,5,7)
    st=torch.randn(5,9)*.1;t=torch.arange(7)
    cp=dict(spatial_rho=.6,time_correlation=(.95*.7**(t[:,None]-t[None]).abs()+.05*torch.eye(7)).tolist())
    return c,eta,st,cp


def test_all_zero_heads_inherit_identical_gaussian_process_reference():
    assert hasattr(module,'make_process_model'), 'trajectory process models are missing'
    c,eta,st,cp=inputs()
    reference=ops.generate_process(None,'reference',c,eta,st,cp,steps=4)
    for mode in ('full','local','affine'):
        model=module.make_process_model(mode)
        assert model.alpha[0]==1 and model.alpha[100]==0
        actual=ops.generate_process(model,mode,c,eta,st,cp,steps=4)
        torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-6)


def test_active_local_process_keeps_common_gp_paths_when_queries_change():
    assert hasattr(module,'make_process_model'), 'trajectory process models are missing'
    c,eta,st,cp=inputs();As=ops.space_factors(st,cp)[0]
    initial=torch.einsum('qr,brh->bqh',As,eta);subset=torch.tensor([4,1,2])
    for mode in ('local','affine'):
        model=module.make_process_model(mode)
        torch.nn.init.normal_(model.net[-1].weight,std=.01)
        full=ops.generate_process(model,mode,c,eta,st,cp,steps=4,spatial_noise=initial)
        small=ops.generate_process(model,mode,c[:,subset],eta[:,subset],st[subset],cp,steps=4,spatial_noise=initial[:,subset])
        torch.testing.assert_close(small,full[:,subset],atol=3e-6,rtol=3e-6)


def test_affine_velocity_and_discrete_sampler_have_exact_gaussian_maps():
    assert hasattr(module,'make_process_model'), 'affine Gaussian mechanism control is missing'
    c,eta,_,_=inputs();model=module.make_process_model('affine')
    torch.nn.init.normal_(model.net[-1].weight,std=.01)
    other=torch.randn_like(eta);k=torch.tensor([.2,.8]);a=.3
    left=model.denoise(a*eta+(1-a)*other,c,k,None,None)
    right=a*model.denoise(eta,c,k,None,None)+(1-a)*model.denoise(other,c,k,None,None)
    torch.testing.assert_close(left,right,atol=2e-6,rtol=2e-6)
    mean,mapping=model.affine_map(c,steps=4)
    actual=ops.heun0(model,c,eta,steps=4)
    torch.testing.assert_close(actual,mean+torch.einsum('bqij,bqj->bqi',mapping,eta),atol=3e-6,rtol=3e-6)
    counts={mode:sum(p.numel() for p in module.make_process_model(mode).parameters()) for mode in ('full','local','affine')}
    assert counts==dict(full=45639,local=44692,affine=44944)
