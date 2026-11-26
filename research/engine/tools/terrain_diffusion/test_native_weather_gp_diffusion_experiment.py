import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).with_name('native_weather_gp_diffusion_experiment.py')
spec=importlib.util.spec_from_file_location('weather_transport_runner',path)
module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_weather_gp_diffusion_experiment as module


def test_frozen_gp_guard_rejects_unfinished_or_changed_sources_except_explicit_smoke():
    assert hasattr(module,'validate_gp'),'frozen weather GP lineage guard missing'
    protocol=dict(source={'weights':'abc'},code_sha256={'kernel':'def'})
    status=dict(status='COMPLETE_WEATHER_GP',steps=3000)
    module.validate_gp(protocol,status,protocol['source'],protocol['code_sha256'],False)
    for altered in ({'status':'ENGINEERING_SMOKE_COMPLETE','steps':4},{'status':'COMPLETE_WEATHER_GP','steps':4}):
        with pytest.raises(ValueError):module.validate_gp(protocol,altered,protocol['source'],protocol['code_sha256'],False)
    module.validate_gp(protocol,dict(status='ENGINEERING_SMOKE_COMPLETE',steps=4),protocol['source'],protocol['code_sha256'],True)
    with pytest.raises(ValueError):module.validate_gp(protocol,status,{'weights':'changed'},protocol['code_sha256'],False)
    with pytest.raises(ValueError):module.validate_gp(protocol,status,protocol['source'],{'kernel':'changed'},False)


def test_formal_transport_rejects_nonregistered_gp_view_or_seed():
    assert hasattr(module,'validate_transport_gp'),'registered G265/seed contract guard missing'
    module.validate_transport_gp(dict(config=dict(view=265,seed=42)),42)
    with pytest.raises(ValueError):module.validate_transport_gp(dict(config=dict(view=175,seed=42)),42)
    with pytest.raises(ValueError):module.validate_transport_gp(dict(config=dict(view=265,seed=17)),42)
