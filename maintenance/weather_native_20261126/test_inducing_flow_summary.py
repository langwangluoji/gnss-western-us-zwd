import importlib
import importlib.util
import pytest


def module():
    assert importlib.util.find_spec('inducing_flow_summary') is not None,'inducing paired summary is missing'
    return importlib.import_module('inducing_flow_summary')


def test_supplement_trigger_uses_registered_frequency_nonlinearity():
    m=module();rows=[dict(name=n,records=dict(crps_h6=dict(value=v))) for n,v in
        [('gaussian_time',8),('affine_frequency',7),('odd_frequency',7.1),('odd_time',6)]]
    assert m.should_supplement(rows) is False
    rows[2]['records']['crps_h6']['value']=6.9
    assert m.should_supplement(rows) is True


def test_sample_layout_and_device_are_part_of_pairing_contract():
    m=module();lin=dict(family='odd',domain='frequency',members=32,sample_seed=261126,
        layout_training_seed=42,device='cuda',independent_iid_member_noise=True,coupled_gp_and_latent_noise=True)
    m.require_pairing(lin,'odd','frequency',42,261126,'cuda')
    for key,value in [('sample_seed',1),('layout_training_seed',17),('device','cpu'),('members',16),
        ('coupled_gp_and_latent_noise',False),('independent_iid_member_noise',False)]:
        with pytest.raises(ValueError):m.require_pairing({**lin,key:value},'odd','frequency',42,261126,'cuda')


def test_supplement_rejects_stale_or_incomplete_primary_summary():
    m=module();names=['gaussian_time','affine_time','affine_frequency','odd_time','odd_frequency']
    rows=[dict(seed=s,name=n,records=dict(crps_h6=dict(value=6.9 if n=='odd_frequency' else 7))) for s in (42,17) for n in names]
    summary=dict(rows=rows,calendar_day=8,lock_sha256='lock',code_sha256='code',supplement_trigger=True)
    m.require_supplement(summary,'lock','code')
    for changed in [{'calendar_day':15},{'lock_sha256':'old'},{'code_sha256':'old'},{'rows':rows[:-1]},
        {'supplement_trigger':False},{'rows':rows[:-1]+[rows[0]]}]:
        with pytest.raises(ValueError):m.require_supplement({**summary,**changed},'lock','code')
