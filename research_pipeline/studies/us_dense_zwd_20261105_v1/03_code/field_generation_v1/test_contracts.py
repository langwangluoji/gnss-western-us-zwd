import unittest
import numpy as np
import torch
from field_generation_v1.geometry import mosaic, query_grid, curvature

class GeometryTests(unittest.TestCase):
    def test_mosaic_constant_and_all_gradients(self):
        patches=torch.ones(4,1,24,24,requires_grad=True)
        origins=np.array([[0,0],[0,12],[12,0],[12,12]])
        result=mosaic(patches,origins,(36,36))
        torch.testing.assert_close(result,torch.ones_like(result))
        query_grid(result,torch.tensor([[17.3,18.2]])).sum().backward()
        self.assertTrue((patches.grad.abs().sum((1,2,3))>0).all())
    def test_linear_query_and_curvature(self):
        y,x=torch.meshgrid(torch.arange(36.),torch.arange(36.),indexing='ij')
        field=(2*x+3*y+10)[None]
        torch.testing.assert_close(query_grid(field,torch.tensor([[12.2,14.4]])),torch.tensor([[77.6]]))
        self.assertEqual(curvature(field,torch.ones(36,36,dtype=torch.bool),1).item(),0)

if __name__=='__main__': unittest.main()
