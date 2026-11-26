import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).with_name('centered_artifact_audit.py')
spec=importlib.util.spec_from_file_location('centered_artifact',path);module=importlib.util.module_from_spec(spec)
if path.exists():spec.loader.exec_module(module)


def test_centered_lineage_rejects_wrong_recipe_for_reference_and_model():
    assert hasattr(module,'validate_lineage'),'centered probability recipe guard missing'
    source={'gp':'frozen'};codes={'scoring':'abc'}
    line=dict(source=source,code_sha256=codes,cache_manifest_sha256='cache8',embedding_manifest_sha256='emb8',
        mode='reference',members=32,seed=261126,independent_iid_member_noise=True,query_count=137)
    module.validate_lineage(line,source,'cache8','emb8','reference',codes)
    for change in ({'source':{'gp':'other'}},{'members':2},{'seed':17},{'mode':'shape'},{'cache_manifest_sha256':'cache15'},{'code_sha256':{'scoring':'other'}}):
        with pytest.raises(ValueError):module.validate_lineage({**line,**change},source,'cache8','emb8','reference',codes)


def test_centered_joint_guard_rejects_stale_lineage_and_scoring():
    assert hasattr(module,'validate_joint'),'joint source binding missing'
    row=dict(source_lineage_sha256='lineage',code_sha256='jointcode',members=32,within_forecast_increment_fair_crps_mm=2.)
    module.validate_joint(row,'lineage','jointcode',2.)
    for change in ({'source_lineage_sha256':'old'},{'code_sha256':'old'},{'members':2}):
        with pytest.raises(ValueError):module.validate_joint({**row,**change},'lineage','jointcode',2.)
