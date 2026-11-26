import unittest
import torch
from field_generation_v1.model import ResidualField,gaussian_losses

class ModelTests(unittest.TestCase):
    def test_faithful_mean_gradient(self):
        feature=torch.randn(3,32,requires_grad=True)
        mean=feature[:,0];target=torch.ones(3)
        head=torch.nn.Linear(32,1)
        q=torch.nn.functional.softplus(head(feature.detach()).squeeze()).square()+1e-4
        mse,nll=gaussian_losses(mean,target,q,1,True)
        expected=torch.autograd.grad(mse,feature,retain_graph=True)[0]
        actual=torch.autograd.grad(mse+nll,feature)[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    def test_variable_tokens_zero_mean(self):
        torch.manual_seed(42);m=ResidualField()
        b=dict(stations=torch.randn(2,7,17),padding=torch.zeros(2,7,dtype=torch.bool),
            coarse=torch.randn(2,4,15,6,6),maps=torch.randn(2,6,24,24),
            baseline_condition=torch.randn(2,1,24,24),mu_condition=torch.randn(2,1,24,24),domain=torch.ones(2,1,24,24))
        r,f=m(b)
        self.assertEqual(tuple(f.shape),(2,32,24,24));self.assertEqual(r.abs().max().item(),0)
        b['padding'][:]=True
        r,f=m(b);self.assertTrue(torch.isfinite(f).all())

if __name__=='__main__': unittest.main()
