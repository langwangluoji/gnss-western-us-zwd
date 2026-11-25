import importlib.util
import json
from pathlib import Path
import pytest
import torch
from .data import sha

path=Path(__file__).with_name('native_model_seal.py')
spec=importlib.util.spec_from_file_location('model_seal',path);module=importlib.util.module_from_spec(spec)
if path.exists():
    from . import native_model_seal as module


def fixture(tmp_path):
    weights={'a':torch.tensor([1.,2.])};torch.save(weights,tmp_path/'last.pt')
    torch.save(dict(step=3000,model=weights),tmp_path/'checkpoint_3000.pt')
    (tmp_path/'status.json').write_text(json.dumps(dict(status='COMPLETE_WEATHER_GP',steps=3000)))
    (tmp_path/'protocol.json').write_text(json.dumps(dict(source={'weights':'frozen'},code_sha256={'kernel':'abc'})))


def test_release_seal_rejects_weight_replacement_and_nonfinal_checkpoint(tmp_path):
    assert hasattr(module,'seal_completed_model'),'completed-model artifact seal missing'
    fixture(tmp_path);module.seal_completed_model(tmp_path,'COMPLETE_WEATHER_GP',{'kernel':'abc'})
    module.validate_model_seal(tmp_path,'COMPLETE_WEATHER_GP',{'kernel':'abc'})
    torch.save({'a':torch.tensor([8.,9.])},tmp_path/'last.pt')
    with pytest.raises(ValueError,match='changed'):module.validate_model_seal(tmp_path,'COMPLETE_WEATHER_GP',{'kernel':'abc'})
    (tmp_path/'model_artifact_seal.json').unlink()
    with pytest.raises(ValueError,match='checkpoint'):module.seal_completed_model(tmp_path,'COMPLETE_WEATHER_GP',{'kernel':'abc'})


def test_evaluation_marker_requires_arrays_metrics_and_final_lineage(tmp_path):
    assert hasattr(module,'evaluation_marker'),'complete evaluation artifact marker missing'
    assert module.evaluation_marker(tmp_path/'absent') is None
    folder=tmp_path/'eval';folder.mkdir();(folder/'metrics.json').write_text('{}')
    with pytest.raises(ValueError,match='incomplete'):module.evaluation_marker(folder)
    (folder/'evaluation.npz').write_bytes(b'example')
    (folder/'lineage.json').write_text(json.dumps(dict(prediction_sha256=sha(folder/'evaluation.npz'),probability_weights_sha256='weights',model_protocol_sha256='protocol')))
    assert module.evaluation_marker(folder)==folder/'lineage.json'
    with pytest.raises(ValueError,match='model'):module.evaluation_marker(folder,{'weights_sha256':'wrong','protocol_sha256':'protocol'})
