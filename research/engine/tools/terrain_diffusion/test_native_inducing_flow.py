import importlib
import importlib.util
import torch


def module():
    name=__package__+'.native_inducing_flow'
    assert importlib.util.find_spec(name) is not None, 'inducing innovation flow is missing'
    return importlib.import_module(name)


def fixture():
    from .native_weather_gp import WeatherProjectiveGP
    torch.manual_seed(12)
    gp=WeatherProjectiveGP(265,dict(spatial_rho=.5,time_correlation=torch.eye(7).tolist())).eval()
    gp.requires_grad_(False)
    c=torch.randn(2,5,175);history=torch.randn(2,72,4,15);history[...,3:6]=1
    static=torch.randn(5,9)*.1;ids=torch.arange(5)
    anchors=torch.tensor([[-1.,0.,0.],[1.,0.,1.],[0.,1.,.5]])
    return gp,c,history,static,ids,anchors


def test_exact_gp_covariance_split_and_query_subset_law():
    m=module();gp,c,history,static,ids,anchors=fixture()
    R,W,residual=m.inducing_decomposition(gp,c,history,static,ids,anchors)
    torch.testing.assert_close(residual+W@W.transpose(-1,-2),R,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(R,gp.correlation(c,history,static,ids),rtol=1e-10,atol=1e-10)
    torch.linalg.cholesky(residual)
    selected=torch.tensor([4,1]);flat=(selected[:,None]*7+torch.arange(7)[None]).flatten()
    r,w,s=m.inducing_decomposition(gp,c[:,selected],history,static[selected],ids[selected],anchors)
    torch.testing.assert_close(r,R[:,flat][:,:,flat],rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(w,W[:,flat],rtol=1e-6,atol=1e-6)


def test_inducing_path_preserves_gp_at_warm_start_and_activates_flow_gradient():
    m=module();torch.manual_seed(8);K,B,Q,D=8,2,3,8
    eta=torch.randn(K,B,D,7);basis=torch.randn(B,Q*7,D*7)*.03
    gp_path=torch.randn(K,B,Q,7)+torch.einsum('bnd,kbd->kbn',basis,eta.flatten(2)).reshape(K,B,Q,7)
    args=dict(condition=torch.randn(B,Q,265),global_condition=torch.randn(B,90),mean=torch.zeros(B,Q,7),
        std=torch.ones(B,Q,7),gp_path=gp_path,latent_noise=eta,basis=basis)
    model=m.InducingLatentFlow('odd','frequency');output=model(**args)
    torch.testing.assert_close(output,gp_path,rtol=0,atol=0)
    output.square().mean().backward()
    assert model.flow.output.weight.grad.abs().sum()>0


def test_geometry_only_anchors_are_deterministic_and_unique():
    m=module();static=torch.randn(30,9)
    a=m.anchor_locations(static,8);b=m.anchor_locations(static,8)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert len(torch.unique(a,dim=0))==8


def test_active_inducing_model_preserves_subset_and_odd_law_with_weather_kernel():
    m=module();gp,c,history,static,ids,anchors=fixture()
    with torch.no_grad():
        gp.node[-1].weight.normal_(0,.005);gp.time[-1].weight.normal_(0,.005)
    R,W,residual=m.inducing_decomposition(gp,c,history,static,ids,anchors)
    K,B,Q,D=4,2,5,3;eta=torch.randn(K,B,D,7)
    noise=torch.randn(K,B,Q*7,dtype=torch.float64)
    path=(torch.einsum('bij,kbj->kbi',torch.linalg.cholesky(residual),noise)+
        torch.einsum('bnd,kbd->kbn',W,eta.double().flatten(2))).reshape(K,B,Q,7).float()
    model=m.InducingLatentFlow('odd','frequency',modes=D)
    with torch.no_grad():
        model.flow.output.weight.normal_(0,.01);model.query.net[-1].weight.normal_(0,.01)
    b=dict(condition=torch.randn(B,Q,265),global_condition=torch.randn(B,90),mean=torch.randn(B,Q,7),
        std=torch.ones(B,Q,7),gp_path=path,latent_noise=eta,basis=W.float())
    full=model(**b);selected=torch.tensor([4,1]);flat=(selected[:,None]*7+torch.arange(7)[None]).flatten()
    small={**b,**{n:b[n][:,selected] for n in ['condition','mean','std']},
        'gp_path':path[:,:,selected],'basis':W[:,flat].float()}
    torch.testing.assert_close(model(**small),full[:,:,selected],rtol=1e-5,atol=2e-5)
    opposite={**b,'gp_path':-path,'latent_noise':-eta}
    torch.testing.assert_close((full+model(**opposite))/2,b['mean'][None].expand_as(full),rtol=0,atol=2e-5)
    r,w,s=m.inducing_decomposition(gp,c[:,selected],history,static[selected],ids[selected],anchors)
    torch.testing.assert_close(r,R[:,flat][:,:,flat],rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(w,W[:,flat],rtol=1e-6,atol=1e-6)
