import importlib.util
from pathlib import Path


def test_probability_registry_covers_full_frequency_tokens_and_equal_solver_calls():
    spec=importlib.util.spec_from_file_location('native_campaign',Path(__file__).with_name('campaign.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert hasattr(module,'probability_variants'), 'probability phases need an explicit registered comparison'
    variants=module.probability_variants()
    by_name={name:(domain,preconditioned,solvers) for name,domain,preconditioned,solvers in variants}
    assert by_name['frequency_tokens']==('frequency_tokens',False,('ddim',))
    assert by_name['preconditioned_frequency_tokens']==('frequency_tokens',True,('ddim','heun'))
    assert by_name['preconditioned_joint']==('joint',True,('ddim','heun'))
    assert len(by_name)==7
    for _,_,preconditioned,solvers in variants:
        if preconditioned:assert solvers==('ddim','heun')
