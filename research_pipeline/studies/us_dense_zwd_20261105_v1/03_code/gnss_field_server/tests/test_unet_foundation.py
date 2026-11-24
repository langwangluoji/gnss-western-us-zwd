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

def test_hybrid_keeps_point_features_and_query_independence():
    from tools.terrain_diffusion.unet_hybrid import Hybrid
    torch.set_num_threads(4);torch.manual_seed(13004);model=Hybrid();point=torch.randn(9,113);grid=torch.randn(1,113,16,16);uv=torch.rand(1,1,9,2)*2-1;hg=torch.zeros(1,1,16,16);hq=torch.rand(9)
    out=model(point,grid,uv,hg,hq);torch.testing.assert_close(out,point[:,52])
    with torch.no_grad():model.spatial.head.bias.fill_(.25);hg.fill_(1.)
    out=model(point,grid,uv,hg,hq);torch.testing.assert_close(out,point[:,52]+.25*torch.exp((1.-hq)/3),atol=1e-6,rtol=1e-6)
    split=torch.cat([model(point[:4],grid,uv[:,:,:4],hg,hq[:4]),model(point[4:],grid,uv[:,:,4:],hg,hq[4:])]);torch.testing.assert_close(out,split)
    out.square().mean().backward();assert torch.isfinite(model.point.net[-1].weight.grad).all();assert torch.isfinite(model.spatial.head.weight.grad).all()
