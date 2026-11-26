import numpy as np
import pytest
import torch


def test_p14_batched_readout_matches_individual_frames():
    from .fixed_support import FixedSupportModel
    torch.manual_seed(71)
    net=FixedSupportModel()
    with torch.no_grad():net.head.weight.normal_(0,.1)
    point=torch.randn(3,7,113)
    support=torch.randn(3,30,8,8)
    uv=torch.rand(3,1,7,2)*2-1
    expected=torch.stack([net(point[i],support[i:i+1],uv[i:i+1]) for i in range(3)])
    torch.testing.assert_close(net(point,support,uv),expected,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('kind',['features','mlp','gru','lstm','transformer'])
def test_new_branches_start_as_exact_p14(kind):
    from .p14_fusion import P14Fusion
    from .fixed_support import FixedSupportModel
    base=FixedSupportModel()
    net=P14Fusion(kind,length=12)
    net.base.load_state_dict(base.state_dict())
    point=torch.randn(2,5,113);support=torch.randn(2,30,8,8)
    uv=torch.rand(2,1,5,2)*2-1
    batch=dict(point=point,support=support,uv=uv,static=torch.randn(2,5,9),
        edges=torch.randn(2,64,8,10),edge_mask=torch.ones(2,64,8),
        grid_static=torch.randn(64,8),shape=(8,8),history=torch.randn(2,12,64,25),
        time_uv=uv[:,0])
    net.eval()
    torch.testing.assert_close(net(batch),base(point,support,uv),atol=0,rtol=0)


def test_chronological_schedule_covers_all_training_hours_without_gaps():
    from .p14_fusion import chronological_blocks
    blocks=chronological_blocks(100,149,12)
    assert [b.tolist() for b in blocks]==[list(range(100,112)),list(range(112,124)),
        list(range(124,136)),list(range(136,148)),[148]]
    np.testing.assert_array_equal(np.concatenate(blocks),np.arange(100,149))


@pytest.mark.real_data
def test_real_continuous_adapter_outputs_finite_float32():
    from .p14_fusion import FusionData
    from .data import PROJECT
    path=PROJECT/'results/20261125_temporal_continuous_data'
    if not torch.cuda.is_available() or not path.exists():pytest.skip('local CUDA/data required')
    data=FusionData(path,length=12)
    b=data.batch(np.arange(100,102),data.events.layouts[0],False,True,True)
    assert b['edges'].shape[-1]==10
    for key in ('point','support','history','edges'):
        assert b[key].dtype==torch.float32 and torch.isfinite(b[key]).all()


@pytest.mark.real_data
def test_same_hour_history_is_independent_of_batch_boundary():
    from .p14_fusion import FusionData
    from .data import PROJECT
    path=PROJECT/'results/20261125_temporal_continuous_data'
    if not torch.cuda.is_available() or not path.exists():pytest.skip('local CUDA/data required')
    data=FusionData(path,length=72)
    c=np.sort(np.random.default_rng(42+28011).choice(410,82,False))
    whole=data.batch(np.arange(71,83),c,False,True,False)
    alone=data.batch(np.array([82]),c,False,True,False)
    torch.testing.assert_close(whole['history'][-1],alone['history'][0],atol=1e-6,rtol=1e-6)


def test_failed_predictions_cannot_disappear_from_scoring():
    from .p14_fusion_experiment import reconstruction_scores
    pred=np.ones((3,1,72,2));pred[:,:,:,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        reconstruction_scores(pred,np.zeros((1,72,2)))
