import numpy as np
import pytest
from retrain_common import ROOT


def test_ridge_field_matches_original_normalized_readout():
    from retrain_geometry import ridge_field
    rng=np.random.default_rng(42);x=rng.normal(size=(7,93)).astype('f4')
    center=rng.normal(size=93);std=rng.uniform(.1,2,93);beta=rng.normal(size=94)
    row=dict(center=center.tolist(),std=std.tolist(),beta=beta.tolist())
    expected=12*((x.astype('f8')-center)/std@beta[:-1]+beta[-1])
    assert np.max(abs(ridge_field(x,row,12)-expected))<1e-10
    with pytest.raises(ValueError):ridge_field(x[:,:92],row,12)
    bad=x.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError):ridge_field(bad,row,12)


def test_moment_aggregation_is_target_subset_invariant():
    from retrain_geometry import MomentOperator
    rng=np.random.default_rng(4);axy=rng.normal(size=(9,2))*1e5;ah=rng.uniform(0,2000,9)
    xy=rng.normal(size=(30,2))*1e5;h=rng.uniform(0,2000,30);r=rng.normal(size=9).astype('f4');valid=rng.random(9)>.2
    full=MomentOperator(xy,h,axy,ah)(r,valid)
    subset=MomentOperator(xy[[3,9,20]],h[[3,9,20]],axy,ah)(r,valid)
    assert np.allclose(full[[3,9,20]],subset,atol=1e-6)
    assert np.array_equal(MomentOperator(xy,h,axy,ah)(r,np.zeros(9,bool)),np.zeros((30,24)))


def test_prior_bundle_identity_changes_with_content(tmp_path):
    from retrain_data import prior_identity
    (tmp_path/'spatial.npz').write_bytes(b'original');(tmp_path/'manifest.json').write_bytes(b'original manifest')
    first=prior_identity(tmp_path);(tmp_path/'spatial.npz').write_bytes(b'different')
    assert prior_identity(tmp_path)!=first


def test_runtime_guard_cannot_exceed_one_card_hour():
    from train_new_base import allowed_runtime
    assert allowed_runtime(3420,1000,1010)==3410
    with pytest.raises(ValueError):allowed_runtime(4000,1000,1010)
    with pytest.raises(ValueError):allowed_runtime(-1,1000,1010)
    assert allowed_runtime(100,1000,1200)==0


def test_checkpoint_rng_stays_on_cpu(tmp_path):
    import torch
    from train_new_base import read_checkpoint
    p=tmp_path/'state.pt';torch.save(dict(rng={'torch':torch.get_rng_state()}),p)
    state=read_checkpoint(p);assert state['rng']['torch'].device.type=='cpu'
    torch.set_rng_state(state['rng']['torch'])


def test_completed_point_checkpoint_is_verified(tmp_path):
    from train_new_base import reuse_completed
    from gnss_field.artifacts import write_json,sha256
    for n in ('joint.pt','point.pt','joint_predictions.npz','point_predictions.npz'):(tmp_path/n).write_bytes(n.encode())
    cfg={'x':1};write_json(tmp_path/'lock.json',cfg)
    products={n:dict(selected={'checkpoint':n+'.pt'},checkpoint_sha256=sha256(tmp_path/(n+'.pt')),prediction_sha256=sha256(tmp_path/(n+'_predictions.npz'))) for n in ('point','joint')}
    write_json(tmp_path/'result.json',dict(checkpoint='joint.pt',checkpoint_sha256=sha256(tmp_path/'joint.pt'),products=products))
    assert reuse_completed(tmp_path,cfg)
    (tmp_path/'point.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='point'):reuse_completed(tmp_path,cfg)


def test_server_attempts_share_one_hour_budget():
    from server_submit import remaining_budget
    assert remaining_budget([60,100])==3440
    assert remaining_budget([3601])==0
    with pytest.raises(ValueError):remaining_budget([-1])


def test_deterministic_training_is_explicit():
    import torch
    from train_new_base import configure_determinism
    original=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False);configure_determinism();assert torch.are_deterministic_algorithms_enabled()
    finally:torch.use_deterministic_algorithms(original)


def test_resume_float_tolerance_rejects_material_difference():
    import torch
    from train_new_base import compare_resume
    assert compare_resume({'x':torch.zeros(3)},{'x':torch.full((3,),1e-8)})<1e-7
    with pytest.raises(AssertionError):compare_resume({'x':torch.zeros(3)},{'x':torch.full((3,),1e-3)})
    with pytest.raises(AssertionError):compare_resume({'x':torch.zeros(3,dtype=torch.uint8)},{'x':torch.ones(3,dtype=torch.uint8)})


def test_code_identity_does_not_use_output_directory(tmp_path,monkeypatch):
    import train_new_base as module
    from gnss_field.artifacts import sha256
    from pathlib import Path
    monkeypatch.setattr(module,'ROOT',tmp_path)
    assert module.data_code_hash()==sha256(Path(module.__file__).with_name('retrain_data.py'))


def test_only_terminal_accounting_releases_retry_budget():
    from server_submit import terminal_accounting
    assert terminal_accounting('COMPLETED') and terminal_accounting('CANCELLED by 1')
    assert not terminal_accounting('RUNNING') and not terminal_accounting('COMPLETING')
