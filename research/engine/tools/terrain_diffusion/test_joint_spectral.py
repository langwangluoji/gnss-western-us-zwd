"""Scientific contracts for the implemented fusion/innovation experiment family."""
import numpy as np
import pytest
import torch


def test_fixed_anchor_identity_missing_data_and_query_invariance():
    from .spectral_operators import AnchorField
    xy=np.array([[0.,0.],[1,0],[0,1],[1,1]])
    field=AnchorField(xy,neighbors=2)
    np.testing.assert_allclose(field.readout(xy),np.eye(4),atol=1e-12)
    q=np.array([[.2,.3],[.8,.4]])
    np.testing.assert_allclose(field.readout(q)[0],field.readout(q[:1])[0])
    truth=np.array([[1.,2.,np.nan,4.]])
    a=field.encode(truth,field.readout(xy),ridge=.001,smooth=0)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a[0,[0,1,3]],truth[0,[0,1,3]]/1.001)
    np.testing.assert_allclose(field.U@field.U.T,np.eye(4),atol=1e-12)
    with pytest.raises(ValueError):field.encode(truth,field.readout(xy),ridge=0)


def test_training_only_standardization_and_ar_inverse():
    from .spectral_operators import InnovationTransform
    rng=np.random.default_rng(52);x=rng.normal(size=(12,6,8))
    op=InnovationTransform.fit(x)
    np.testing.assert_allclose(op.inverse(op.forward(x)),x,atol=1e-12)
    assert np.abs(op.rho).max()<=.95
    assert (op.scale>=1e-4).all()
    first=op.forward(x)
    changed=x.copy();changed[:,3:]+=100
    np.testing.assert_allclose(op.forward(changed)[:,:3],first[:,:3])


def test_history_spectrum_obeys_origin_and_missing_masks():
    from .spectral_operators import history_spectrum
    t=np.arange(100);y=np.sin(t*2*np.pi/12)[:,None]
    y[10:14]=np.nan
    result=history_spectrum(y,71,length=72)
    other=y.copy();other[72:]+=1e5
    np.testing.assert_array_equal(result,history_spectrum(other,71,length=72))
    assert np.isfinite(result).all()
    # A constant signal has zero nonzero-frequency amplitude even with gaps.
    constant=history_spectrum(np.ones((72,1)),71).reshape(1,4,4)
    assert np.max(np.abs(constant[:,:,:3]))<1e-6
    np.testing.assert_equal(constant[:,:,3],1.)


def test_learned_label_representation_is_set_invariant_and_linear_in_latent():
    from .joint_probability import LatentField
    torch.manual_seed(6);model=LatentField(dimension=16,condition_width=8)
    u=torch.randn(2,7);static=torch.randn(7,9);mask=torch.ones(2,7,dtype=torch.bool)
    z=model.encode(u,static,mask)
    order=torch.randperm(7)
    torch.testing.assert_close(z,model.encode(u[:,order],static[order],mask[:,order]))
    q=torch.randn(2,5,18)
    torch.testing.assert_close(model.decode(2*z,q),2*model.decode(z,q))


def test_exact_window_causal_encoder_and_zero_initialized_fusion():
    from .masked_transformer_fusion import CausalSpatialEncoder,MaskedP14Fusion
    torch.manual_seed(9);torch.set_num_threads(2)
    enc=CausalSpatialEncoder(channels=25,width=16,side=4,layers=2).eval()
    x=torch.randn(2,8,16,25);static=torch.randn(16,8)
    out=enc(x,static,all_times=True)
    changed=x.clone();changed[:,4:]+=100
    torch.testing.assert_close(enc(changed,static,all_times=True)[:,:4],out[:,:4])
    torch.testing.assert_close(enc(x[:1],static),enc(x,static)[:1],atol=1e-6,rtol=1e-5)
    model=MaskedP14Fusion('T5',length=8,side=4,width=16).eval()
    b=dict(point=torch.randn(2,5,113),support=torch.randn(2,30,8,8),
        uv=torch.rand(2,1,5,2)*2-1,shape=(8,8),static=torch.randn(2,5,9),
        grid_static=torch.randn(64,8),edges=torch.randn(2,64,3,10),
        edge_mask=torch.ones(2,64,3),history=x,time_static=static,
        time_uv=torch.rand(2,5,2)*2-1,calendar=torch.randn(2,4))
    b['fusion_uv']=torch.rand(2,8,8,2)*2-1
    torch.testing.assert_close(model(b),model.reference(b),atol=0,rtol=0)


@pytest.mark.parametrize('domain',['spatial','graph','dual'])
def test_shared_state_denoiser_has_gradient_and_query_independent_sampling(domain):
    from .joint_probability import InnovationDiffusion
    torch.manual_seed(81)
    U=torch.linalg.qr(torch.randn(8,8)).Q
    model=InnovationDiffusion(8,condition_dim=12,domain=domain,U=U,width=32)
    x=torch.randn(3,8);c=torch.randn(3,12)
    loss=model.loss(x,c);loss.backward()
    assert torch.isfinite(loss) and sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)>0
    model.eval();noise=torch.randn(3,8)
    a=model.sample(c,noise,steps=4)
    b=model.sample(c[:1],noise[:1],steps=4)
    torch.testing.assert_close(a[:1],b,atol=1e-5,rtol=1e-5)


def test_future_joint_dct_inverse_and_origin_contract():
    from .spectral_operators import joint_transform,require_forecast_origin
    x=np.random.default_rng(8).normal(size=(2,6,8));U=np.linalg.qr(np.random.default_rng(9).normal(size=(8,8)))[0]
    np.testing.assert_allclose(joint_transform(joint_transform(x,U),U,inverse=True),x,atol=1e-12)
    require_forecast_origin(np.array([12,12]),np.array([12,12]),np.array([12,12]))
    with pytest.raises(ValueError):require_forecast_origin(np.array([12]),np.array([13]),np.array([12]))


def test_split_window_rejects_boundary_and_fold_groups():
    from .joint_data import validate_window,fold_groups
    hours=np.arange(100)
    validate_window(hours,np.arange(20,24),length=12,start=9,stop=80)
    with pytest.raises(ValueError):validate_window(hours,np.arange(19,23),length=12,start=9,stop=80)
    with pytest.raises(ValueError):validate_window(hours,np.array([20,22]),length=12,start=9,stop=80)
    groups=np.repeat(np.arange(10),2);folds=fold_groups(groups)
    assert len(folds)==5
    assert sorted(np.concatenate(folds).tolist())==list(range(20))
    for f in folds:assert not set(groups[f])&set(groups[np.setdiff1d(np.arange(20),f)])


def test_probability_gate_and_fair_crps():
    from .joint_experiment import enforce_representation_gate,fair_crps,forecast_targets
    with pytest.raises(RuntimeError,match='representation'):
        enforce_representation_gate(1.1,diagnostic=False)
    enforce_representation_gate(1.1,diagnostic=True)
    np.testing.assert_allclose(fair_crps(np.array([[-1.],[1.]]),np.array([0.])),0.)
    values=np.arange(100*3).reshape(100,3).astype(float)
    target,mean=forecast_targets(values,origin=20,mean_at_origin=np.array([2.,3.,4.]))
    np.testing.assert_equal(target,values[21:27]);np.testing.assert_equal(mean,np.tile([2.,3.,4.],(6,1)))


def test_band_whitening_is_single_state_with_invertible_noise_coordinates():
    from .joint_probability import InnovationDiffusion
    torch.manual_seed(3);U=torch.linalg.qr(torch.randn(8,8)).Q
    scale=torch.linspace(.2,2.,8);M=torch.diag(scale)@U.T
    net=InnovationDiffusion(8,12,domain='dual',U=U,width=32,state_to_space=M)
    x=torch.randn(2,8)
    torch.testing.assert_close(x@net.state_to_space@net.space_to_state,x,atol=1e-5,rtol=1e-5)
    loss=net.loss(x,torch.randn(2,12));loss.backward();assert torch.isfinite(loss)


def test_past_spectral_condition_does_not_treat_missing_support_as_zero_observation():
    from .joint_experiment import past_spectral_condition
    h=np.zeros((72,4,26),dtype='float32')
    out=past_spectral_condition(h)
    np.testing.assert_array_equal(out,np.zeros(16))
    h[:,0,3]=1;h[:,0,0]=np.cos(np.arange(72)*2*np.pi/12)
    first=past_spectral_condition(h)
    h[:,1:,0]=1e5 # coverage remains zero, values must not enter LS
    np.testing.assert_array_equal(first,past_spectral_condition(h))
    np.testing.assert_allclose(first.reshape(4,4)[:,3],.25)


def test_temporal_to_dense_uses_physical_coordinates_instead_of_tensor_extent():
    from .masked_transformer_fusion import resample_support
    x=torch.linspace(-1,1,4);field=x[None,None,None,:].expand(1,1,4,4)
    uv=torch.tensor([[[[-.3,0.],[.7,.2]],[[1.5,0.],[-2.,0.]]]])
    expected=torch.tensor([[[[-.3,.7],[1.,-1.]]]])
    torch.testing.assert_close(resample_support(field,uv),expected,atol=1e-6,rtol=1e-6)


def test_denoiser_cross_attention_uses_physical_memory_at_every_layer():
    from .joint_probability import InnovationDiffusion
    net=InnovationDiffusion(16,1027,width=32)
    loss=net.loss(torch.randn(2,16),torch.randn(2,1027));loss.backward()
    assert len(net.space.cross_layers)==2
    for layer in net.space.cross_layers:
        assert layer.multihead_attn.in_proj_weight.grad.abs().sum()>0


def test_upstream_lineage_cannot_silently_relabel_a_pilot_as_formal(tmp_path):
    from .joint_experiment import verify_lineage
    from .data import write_json
    write_json(tmp_path/'protocol_foundation.json',dict(config=dict(diagnostic=True,steps=60),code_sha256={},data_sha256='a'))
    with pytest.raises(ValueError,match='diagnostic'):
        verify_lineage(tmp_path,False,'a',{})
    with pytest.raises(ValueError,match='data'):
        verify_lineage(tmp_path,True,'b',{})


def test_incomplete_oof_cache_is_rejected_before_representation_fit(tmp_path):
    from .joint_experiment import load_events
    (tmp_path/'fold0').mkdir()
    with pytest.raises(ValueError,match='five'):
        load_events(tmp_path,'train')


def test_test_collection_keeps_standard_library_profile_unshadowed():
    import importlib.util
    spec=importlib.util.find_spec('profile')
    assert 'terrain_diffusion' not in spec.origin


def test_velocity_prediction_has_exact_clean_noise_inverse_and_finite_sampling():
    from .velocity_diffusion import VelocityDiffusion,velocity_inverse
    torch.manual_seed(15);x=torch.randn(3,8);eps=torch.randn(3,8);a=torch.tensor([[1e-5],[.2],[.9]])
    noisy=a.sqrt()*x+(1-a).sqrt()*eps;v=a.sqrt()*eps-(1-a).sqrt()*x
    clean,noise=velocity_inverse(noisy,v,a)
    torch.testing.assert_close(clean,x);torch.testing.assert_close(noise,eps)
    net=VelocityDiffusion(8,12,width=32);c=torch.randn(3,12)
    loss=net.loss(x,c);loss.backward();assert torch.isfinite(loss)
    samples=net.sample(c,eps,steps=20)
    assert torch.isfinite(samples).all() and samples.abs().max()<20


def test_representation_gate_uses_the_selected_family_and_spatial_check():
    from .joint_experiment import check_family_gate
    good=dict(reconstruction_train_mm=.5,reconstruction_inner2019_mm=.7,
              masked_label_reconstruction_mm=.8,masked_events=2)
    assert check_family_gate(good,False)['passed']
    bad=dict(good,reconstruction_inner2019_mm=2.)
    with pytest.raises(RuntimeError,match='representation'):check_family_gate(bad,False)
    assert not check_family_gate(bad,True)['passed']


@pytest.mark.real_data
def test_real_split_fold_adapter_query_and_unavailable_values():
    from .joint_data import JointData,fold_groups
    from .data import PROJECT
    from .masked_transformer_fusion import MaskedP14Fusion
    path=PROJECT/'results/20261125_temporal_continuous_data'
    if not torch.cuda.is_available() or not path.exists():pytest.skip('local CUDA/data required')
    groups=np.load(path/'geometry.npz')['groups'];outer=fold_groups(groups)[0]
    keep=np.setdiff1d(np.arange(410),outer)
    data=JointData(path,length=12,side=4,train_ids=keep)
    c=keep[:60];anchors=np.arange(24,26)
    b=data.batch(anchors,c)
    assert b['history'].shape==(2,12,16,25)
    assert not b['mask'][:,outer].any()
    model=MaskedP14Fusion('T5',length=12,side=4,width=16).cuda().eval()
    with torch.no_grad():
        original=model(b)
        saved=data.values[24:27,outer].clone();data.values[24:27,outer]=1e5
        changed=data.batch(anchors,c)
        torch.testing.assert_close(model(changed),original,atol=0,rtol=0)
        data.values[24:27,outer]=saved
        small=data.query_batch(anchors,c,b)
        assert small['point'].shape==(2,16,113)
        assert torch.isfinite(model(small)).all()
