import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).with_name('weather_gp_supplement.py')
spec=importlib.util.spec_from_file_location('extra_gp',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_extra_gp_reuse_guard_binds_cpu_recipe_and_reference_family():
    assert hasattr(module,'validate_extra_lineage'),'extra GP inference recipe guard missing'
    source={'mean':'frozen'};codes={'kernel':'abc'}
    line=dict(source=source,cache_manifest_sha256='cache15',embedding_manifest_sha256='emb15',code_sha256=codes,
        members=32,seed=261126,device='cpu',independent_iid_member_noise=True,probability_weights_sha256=None,model_protocol_sha256=None)
    module.validate_extra_lineage(line,source,'cache15','emb15',codes,True)
    for change in ({'members':2},{'device':'cuda'},{'cache_manifest_sha256':'cache8'},{'probability_weights_sha256':'other'}):
        with pytest.raises(ValueError):module.validate_extra_lineage({**line,**change},source,'cache15','emb15',codes,True)
