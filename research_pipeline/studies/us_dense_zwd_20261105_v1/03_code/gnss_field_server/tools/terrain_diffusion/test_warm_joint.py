import numpy as np
import pytest
import torch


def test_warm_transformer_starts_at_exact_best_reference():
    from .warm_joint_experiment import WarmJoint
    from .p14_fusion import P14Fusion
    torch.manual_seed(71);old=P14Fusion('gru',12)
    with torch.no_grad():old.base.point[-1].weight.normal_(0,.02)
    net=WarmJoint(old.state_dict(),'gru','T5',length=12,side=4,width=16).eval()
    uv=torch.rand(2,1,5,2)*2-1
    b=dict(point=torch.randn(2,5,113),support=torch.randn(2,30,8,8),uv=uv,shape=(8,8),
        static=torch.randn(2,5,9),grid_static=torch.randn(64,8),edges=torch.randn(2,64,3,10),
        edge_mask=torch.ones(2,64,3),history=torch.randn(2,12,16,25),time_static=torch.randn(16,8),
        legacy_history=torch.randn(2,12,64,25),time_uv=uv[:,0],calendar=torch.randn(2,4),fusion_uv=torch.rand(2,8,8,2)*2-1)
    with torch.no_grad():
        reference=dict(b,history=b['legacy_history'])
        torch.testing.assert_close(net(b),old.eval()(reference),atol=0,rtol=0)
    assert not any(p.requires_grad for p in net.reference.parameters())


@pytest.mark.real_data
def test_warm_data_preserves_original_normalization_and_legacy_history():
    from .warm_joint_experiment import WarmData
    from .p14_fusion import FusionData
    from .data import PROJECT
    path=PROJECT/'results/20261125_temporal_continuous_data'
    if not path.exists() or not torch.cuda.is_available():pytest.skip('local CUDA/data required')
    old=FusionData(path,72);data=WarmData(path)
    c=old.events.layouts[0];a=np.arange(82,84)
    original=old.batch(a,c);current=data.batch(a,c)
    for key in ('point','support','edges','edge_mask','target','uv'):
        torch.testing.assert_close(current[key],original[key],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(current['legacy_history'],original['history'],atol=1e-6,rtol=1e-6)
