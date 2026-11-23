import numpy as np
import torch
from tools.terrain_diffusion.unet_foundation import Experiment,GridUNet
from tools.terrain_diffusion.residual_field import features_from_batch

def test_gpu_feature_contract():
    torch.set_num_threads(4);ex=Experiment();e=ex.fit[0];c=e['context'];q=np.arange(410)
    # Use actual disjoint training query subset for the canonical training extractor.
    q=np.flatnonzero(e['target']);d=ex.data.batch(e['row'],12+e['hour'],c,q)
    s,h=features_from_batch(d,ex.data.mean[0],ex.data.std[0]);expected=np.concatenate([s,h[:,-1],h.mean(1),h[:,-1]-h[:,0]],-1)
    actual=ex.get_features(e,'mlp').cpu().numpy()[q]
    np.testing.assert_allclose(actual,expected,atol=.004,rtol=.002)
    # Changing hidden targets leaves actual inputs exactly unchanged.
    before=ex.get_features(e,'unet').clone();ex.raw['train'][e['row'],:,q,0]+=100
    torch.testing.assert_close(before,ex.get_features(e,'unet'),atol=0,rtol=0)

def test_grid_readout_and_gradient():
    torch.set_num_threads(4);ex=Experiment();net=GridUNet(16).cuda();e=ex.fit[0];f=ex.get_features(e,'unet');field=net(f.T.reshape(1,113,*ex.shape))
    torch.testing.assert_close(field.flatten(),f[:,52])
    a=ex.query(field,ex.train_features.geo);b=torch.cat([ex.query(field,g) for g in ex.train_features.geo.split(70)])
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6);a.square().mean().backward();assert torch.isfinite(net.head.weight.grad).all()
