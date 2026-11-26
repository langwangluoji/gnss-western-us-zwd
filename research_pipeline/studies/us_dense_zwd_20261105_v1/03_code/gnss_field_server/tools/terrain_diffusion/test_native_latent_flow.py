import importlib
import importlib.util
import torch


def module():
    name=__package__+'.native_latent_flow'
    assert importlib.util.find_spec(name) is not None, 'projective latent flow is missing'
    return importlib.import_module(name)


def inputs():
    torch.manual_seed(13)
    return dict(condition=torch.randn(2,3,265),global_condition=torch.randn(2,90),
        mean=torch.randn(2,3,7)+100,std=torch.rand(2,3,7)+5,
        gp_path=torch.randn(4,2,3,7),latent_noise=torch.randn(4,2,8,7))


def test_zero_initialization_recovers_exact_gp_for_all_four_models():
    m=module();b=inputs();counts=[]
    for family in ['affine','odd']:
        for domain in ['time','frequency']:
            model=m.ProjectiveLatentFlow(family,domain)
            actual=model(**b);expected=b['mean'][None]+b['std'][None]*b['gp_path']
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(model.latent(b['latent_noise'],b['global_condition']),b['latent_noise'],rtol=0,atol=0)
            counts.append(sum(p.numel() for p in model.parameters()))
    assert max(counts)/min(counts)<1.05


def test_active_model_has_projective_query_decoder_and_exact_central_symmetry():
    m=module();b=inputs();model=m.ProjectiveLatentFlow('odd','frequency')
    with torch.no_grad():
        model.query.net[-1].weight.normal_(0,.02);model.flow.output.weight.normal_(0,.02)
    full=model(**b);ids=torch.tensor([2,0]);small={**b,'condition':b['condition'][:,ids],
        'mean':b['mean'][:,ids],'std':b['std'][:,ids],'gp_path':b['gp_path'][:,:,ids]}
    torch.testing.assert_close(model(**small),full[:,:,ids],rtol=1e-5,atol=2e-5)
    opposite={**b,'gp_path':-b['gp_path'],'latent_noise':-b['latent_noise']}
    torch.testing.assert_close((full+model(**opposite))/2,b['mean'][None].expand_as(full),rtol=0,atol=2e-5)


def test_affine_flow_is_linear_for_fixed_conditions_after_activation():
    m=module();b=inputs();model=m.ProjectiveLatentFlow('affine','time')
    with torch.no_grad():model.flow.net[-1].weight.normal_(0,.01)
    x=b['latent_noise'];z=torch.randn_like(x);c=b['global_condition']
    torch.testing.assert_close(model.latent(x+z,c),model.latent(x,c)+model.latent(z,c),rtol=1e-5,atol=1e-5)


def test_joint_score_masks_missing_labels_without_nan_gradients():
    m=module();b=inputs();samples=torch.randn(4,2,3,7,requires_grad=True)
    truth=torch.randn(2,3,7);truth[0]=float('nan');truth[1,0,2]=float('nan')
    R=torch.eye(21)[None].expand(2,-1,-1);pairs=torch.tensor([[0,1],[1,2]])
    loss,terms=m.joint_score(samples,truth,torch.zeros_like(truth),torch.ones_like(truth),R,pairs)
    assert torch.isfinite(loss);loss.backward();assert torch.isfinite(samples.grad).all()
    assert samples.grad[:,0].abs().sum()==0
    assert all(torch.isfinite(v) for v in terms.values())
