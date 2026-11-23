import copy
import torch
import pytest


def models():
    import pilot_models
    return pilot_models


@pytest.mark.parametrize('variant', ['plain','noise','attention','mp'])
def test_gaussian_initialization_and_live_update(variant):
    m=models(); torch.manual_seed(42)
    net=m.make_generator('gan' if variant in ('plain','noise') else 'diffusion',variant,widths=(8,16,16,16),steps=18)
    c=torch.randn(2,69,16,16);p=torch.randn(2,1,16,16);z=torch.randn_like(p)
    before={k:v.clone() for k,v in net.state_dict().items()}
    y=net.sample_residual(c,p,z)
    assert (y-p).abs().max()<1e-4
    loss=(y-1).square().mean();loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
    assert any(v.grad is not None and v.grad.abs().max()>0 for v in net.parameters())
    assert all(torch.equal(v,before[k]) for k,v in net.state_dict().items())
    opt=torch.optim.Adam(net.parameters(),lr=1e-3);opt.step();opt.zero_grad()
    y=net.sample_residual(c,p,z);(y-1).square().mean().backward()
    if variant=='noise':
        assert any(v.grad is not None and v.grad.abs().sum()>0 for n,v in net.named_parameters() if 'noise_gain' in n)
    if variant=='attention':
        assert net.net.attention.out.weight.grad.abs().sum()>0
    assert (y-p).abs().max()>0


def test_pair_critic_permutation_mask_and_gp():
    from gnss_field.models.grid_generation import critic_gradient_penalty
    m=models();torch.manual_seed(5);critic=m.PairCritic(width=8,max_edges=16)
    r=torch.randn(2,9);xy=torch.randn(9,3);c=torch.randn(2,9,69);mask=torch.ones(2,9,dtype=torch.bool);mask[:,8]=False
    perm=torch.tensor([8,2,5,1,7,0,6,3,4])
    a=critic(r,xy,c,mask);b=critic(r[:,perm],xy[perm],c[:,perm],mask[:,perm])
    assert torch.allclose(a,b,atol=1e-6)
    r[:,8]=float('nan');c[:,8]=float('nan')
    assert torch.allclose(a,critic(r,xy,c,mask),atol=1e-6)
    real=torch.nan_to_num(r).requires_grad_();fake=torch.randn_like(real)
    gp=critic_gradient_penalty(critic,real,fake,xy,torch.nan_to_num(c),mask)
    gp.backward();assert torch.isfinite(gp)
    assert all(torch.isfinite(p.grad).all() for p in critic.parameters() if p.grad is not None)
    assert torch.isfinite(critic(real,xy,torch.nan_to_num(c),torch.arange(9)[None].expand(2,-1)==0)).all()


@pytest.mark.parametrize('attention',[False,True])
def test_set_denoiser_permutation_missing_and_unroll(attention):
    m=models();torch.manual_seed(4);net=m.SetDenoiser(attention=attention,width=16)
    torch.nn.init.normal_(net.output.weight,std=.02)
    x=torch.randn(2,7);c=torch.randn(2,7,69);xy=torch.randn(7,3);mask=torch.ones(2,7,dtype=torch.bool);mask[:,6]=False
    p=torch.randperm(7)
    a=net.denoise(x,torch.tensor(.7),c,xy,mask)
    b=net.denoise(x[:,p],torch.tensor(.7),c[:,p],xy[p],mask[:,p])
    assert torch.allclose(a[:,p],b,atol=1e-6)
    xx=x.clone();xx[:,6]=float('nan');cc=c.clone();cc[:,6]=float('nan')
    assert torch.allclose(a,net.denoise(xx,.7,cc,xy,mask),atol=1e-6)
    from gnss_field.models.diffusion import heun_sample
    y=heun_sample(lambda state,sigma:net.denoise(state,sigma,c,xy,mask),x,domain=mask,steps=18)
    y.square().mean().backward()
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_matched_baseline_paths_and_split_noise():
    from gnss_field.models.diffusion import keyed_noise
    m=models()
    for kind,variant in [('gan','noise'),('diffusion','attention'),('diffusion','mp')]:
        torch.manual_seed(42);base=m.make_generator(kind,widths=(8,16,16,16))
        torch.manual_seed(42);other=m.make_generator(kind,variant,widths=(8,16,16,16))
        shared={k:v for k,v in base.net.state_dict().items() if k in other.net.state_dict() and v.shape==other.net.state_dict()[k].shape}
        if variant=='mp':shared={k:v for k,v in shared.items() if k.startswith(('time_embedding.','film.','frequencies'))}
        assert all(torch.equal(v,other.net.state_dict()[k]) for k,v in shared.items())
    all_noise=keyed_noise([1,2],4,42,shape=(1,16,16))
    assert torch.equal(all_noise[:1],keyed_noise([1],4,42,shape=(1,16,16)))


def test_real_sampler_optimizer_resume():
    m=models();torch.manual_seed(44)
    net=m.make_generator('diffusion','mp',widths=(8,16,16,16),recompute=True)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
    c=torch.randn(2,69,16,16);p=torch.randn(2,1,16,16);z=torch.randn_like(p)
    def update(model,optimizer):
        optimizer.zero_grad();loss=model.sample_residual(c,p,z).square().mean();loss.backward();optimizer.step()
    update(net,opt);saved=copy.deepcopy((net.state_dict(),opt.state_dict()))
    update(net,opt);expected=copy.deepcopy(net.state_dict())
    resumed=m.make_generator('diffusion','mp',widths=(8,16,16,16),recompute=True)
    ropt=torch.optim.AdamW(resumed.parameters(),lr=1e-3);resumed.load_state_dict(saved[0]);ropt.load_state_dict(saved[1]);update(resumed,ropt)
    assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in expected.items())


def test_input_cache_corruption_is_rejected(tmp_path):
    import numpy as np
    from pilot_data import validate_inputs
    from grid_pair_data import row_digest
    from gnss_field.artifacts import write_json,sha256
    cache=tmp_path/'cache';cache.mkdir();source=tmp_path/'source.npz';source.write_bytes(b'fixed source')
    value=np.ones((2,3,4,4),'f4');np.save(cache/'conditions69.npy',value);np.save(cache/'row_hashes.npy',np.stack([row_digest([v]) for v in value]))
    write_json(cache/'cursor.json',dict(done=[True,True]));write_json(cache/'contract.json',dict(source_data=sha256(source)))
    write_json(cache/'complete.json',dict(contract=sha256(cache/'contract.json'),row_hashes=sha256(cache/'row_hashes.npy'),hours=2,complete=True))
    validate_inputs(cache,source)
    value[0,0,0,0]=99;np.save(cache/'conditions69.npy',value)
    with pytest.raises(ValueError,match='checksum'):validate_inputs(cache,source)


def test_completed_arm_reuse_rejects_changed_contract(tmp_path):
    from run_pilot import reuse_result
    from gnss_field.artifacts import write_json,sha256
    checkpoint=tmp_path/'candidate.pt';checkpoint.write_bytes(b'fixed checkpoint')
    metadata=dict(data='a',code='b');write_json(tmp_path/'lock.json',metadata)
    write_json(tmp_path/'result.json',dict(checkpoint=checkpoint.name,checkpoint_sha256=sha256(checkpoint)))
    assert reuse_result(tmp_path,metadata)['checkpoint']==checkpoint.name
    with pytest.raises(ValueError,match='contract'):reuse_result(tmp_path,dict(data='changed',code='b'))
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='checkpoint'):reuse_result(tmp_path,metadata)
