import torch
import pytest

@pytest.mark.parametrize('kind',['stochastic','gan','diffusion'])
def test_gaussian_initialization_is_prior_not_collapsed(kind):
 from gnss_field.models.grid_generation import ResidualGenerator
 torch.manual_seed(42)
 m=ResidualGenerator(kind,widths=(8,16,16,16)).double()
 x=torch.randn(2,69,16,16,dtype=torch.float64)
 prior=torch.randn(2,1,16,16,dtype=torch.float64)*8
 white=torch.randn_like(prior)
 actual=m.sample_residual(x,prior,white)
 torch.testing.assert_close(actual,prior,atol=1e-10,rtol=1e-10)
 assert actual.std()>1


def test_diffusion_35_calls_and_observation_gradient():
 from gnss_field.models.grid_generation import ResidualGenerator
 from gnss_field.models.grid_processor import query_field
 m=ResidualGenerator('diffusion',widths=(8,16,16,16))
 calls=[]; original=m.denoise
 def record(*args):
  calls.append(1)
  return original(*args)
 m.denoise=record
 x=torch.randn(1,69,16,16)
 out=m.sample_residual(x,torch.randn(1,1,16,16),torch.randn(1,1,16,16))
 assert len(calls)==35
 query_field(out,torch.tensor([[0,1,16,17]]),torch.tensor([[.1,.2,.3,.4]])).sum().backward()
 assert len(calls)==70
 g=m.net.output.weight.grad
 assert torch.isfinite(g).all() and g.abs().sum()>0


def test_critic_permutation_padding_and_masked_values():
 from gnss_field.models.grid_generation import ObservationCritic
 torch.manual_seed(2)
 m=ObservationCritic()
 y=torch.randn(2,7); xy=torch.randn(7,3); c=torch.randn(2,7,69)
 mask=torch.ones(2,7,dtype=torch.bool); mask[:,5:]=False
 a=m(y,xy,c,mask)
 order=torch.tensor([6,3,0,5,2,1,4])
 torch.testing.assert_close(a,m(y[:,order],xy[order],c[:,order],mask[:,order]))
 y[:,5:]=float('nan'); c[:,5:]=float('nan')
 torch.testing.assert_close(a,m(y,xy,c,mask))
 with pytest.raises(ValueError): m(y,xy,c,mask*False)


def test_prior_grid_and_station_factors_are_equivalent():
 from gnss_field.models.grid_generation import gaussian_prior_field
 from gnss_field.models.grid_processor import query_field
 torch.manual_seed(2)
 phi=torch.randn(16,7,dtype=torch.float64)
 v=torch.linalg.qr(torch.randn(7,3,dtype=torch.float64)).Q.T
 z=torch.randn(2,4,3,dtype=torch.float64); e=torch.randn(2,4,7,dtype=torch.float64)
 field=gaussian_prior_field(phi,v,z,e,(4,4))
 idx=torch.tensor([[0,1,4,5],[5,6,9,10]])
 w=torch.tensor([[.1,.2,.3,.4],[.2,.3,.1,.4]],dtype=torch.float64)
 actual=query_field(field.flatten(0,1),idx,w).reshape(2,4,2)
 h=(phi[idx]*w[...,None]).sum(1)
 expected=z@(h@v.T).T+e@(h-(h@v.T)@v).T
 torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


def test_keyed_noise_is_stable_across_batch_member_and_query_changes():
 from gnss_field.models.diffusion import keyed_noise
 a=keyed_noise([10,20],4,42,shape=(1,16,16))
 b=keyed_noise([20],7,42,shape=(1,16,16))
 torch.testing.assert_close(a[1],b[0,:4],atol=0,rtol=0)


def test_full_graph_and_recomputed_diffusion_are_numerically_equivalent():
 from gnss_field.models.grid_generation import ResidualGenerator
 torch.set_num_threads(1); torch.manual_seed(42)
 a=ResidualGenerator('diffusion',widths=(8,16,16,16),recompute=True).double()
 with torch.no_grad(): a.net.output.weight.normal_(std=.01)
 b=ResidualGenerator('diffusion',widths=(8,16,16,16),recompute=False).double(); b.load_state_dict(a.state_dict())
 x=torch.randn(1,69,8,8,dtype=torch.float64); prior=torch.randn(1,1,8,8,dtype=torch.float64); white=torch.randn_like(prior)
 ya=a.sample_residual(x,prior,white); yb=b.sample_residual(x,prior,white)
 torch.testing.assert_close(ya,yb,rtol=1e-12,atol=1e-12)
 ya.square().mean().backward(); yb.square().mean().backward()
 for pa,pb in zip(a.parameters(),b.parameters(),strict=True):
  torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-10,atol=1e-10)
