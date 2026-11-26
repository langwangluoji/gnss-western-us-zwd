import importlib.util
import json
from pathlib import Path
import pytest
import torch
from .native_centered_shape import CenteredShape
from .native_model_seal import seal_completed_model

path=Path(__file__).with_name('native_centered_shape_experiment.py')
spec=importlib.util.spec_from_file_location('centered_runner',path);module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_centered_shape_experiment as module


def test_centered_training_combines_data_role_budget_and_registered_gp_guards():
    assert hasattr(module,'training_guard'),'centered training contract missing'
    source=dict(data_sha256='a',source_sha256='b',I1_sha256='c');gp=dict(config=dict(view=265,seed=42))
    module.training_guard(False,3000,3000,False,True,source,source,gp,42)
    for development,steps,view in ((True,3000,265),(False,4,265),(False,3000,175)):
        with pytest.raises(ValueError):module.training_guard(development,3000,steps,False,True,source,source,dict(config=dict(view=view,seed=42)),42)


def test_loading_centered_model_checks_completed_seal_and_source(tmp_path):
    assert hasattr(module,'load_centered_model'),'sealed centered-model loader missing'
    model=CenteredShape('shape');torch.save(model.state_dict(),tmp_path/'last.pt')
    torch.save(dict(step=3000,model=model.state_dict()),tmp_path/'checkpoint_3000.pt')
    protocol=dict(config=dict(mode='shape'),source={'gp':'frozen'},code_sha256=module.codes())
    (tmp_path/'protocol.json').write_text(json.dumps(protocol));(tmp_path/'status.json').write_text(json.dumps(dict(status='COMPLETE_CENTERED_SHAPE',steps=3000)))
    seal_completed_model(tmp_path,'COMPLETE_CENTERED_SHAPE',module.codes())
    loaded,_=module.load_centered_model(tmp_path,{'gp':'frozen'},'cpu')
    c=torch.randn(2,4,265);eta=torch.randn(2,4,7)
    torch.testing.assert_close(loaded.transform(c,eta),eta,atol=0,rtol=0)
    with pytest.raises(ValueError):module.load_centered_model(tmp_path,{'gp':'wrong'},'cpu')
    weights=model.state_dict();weights['net.4.bias']=weights['net.4.bias']+1;torch.save(weights,tmp_path/'last.pt')
    with pytest.raises(ValueError):module.load_centered_model(tmp_path,{'gp':'frozen'},'cpu')
