from types import SimpleNamespace
import json
import numpy as np
import pytest
from . import native_weather_experiment as experiment
from .data import sha
from .native_weather_cache import static


@pytest.mark.parametrize('function',[experiment.train_mean,experiment.train_probability])
def test_development_cache_is_rejected_before_any_output_or_training(tmp_path,monkeypatch,function):
    monkeypatch.setattr(experiment,'CachedData',lambda path:SimpleNamespace(development=True))
    out=tmp_path/'should_not_exist'
    with pytest.raises(ValueError,match='development'):
        function(SimpleNamespace(cache=tmp_path,out=out))
    assert not out.exists()


@pytest.fixture
def locked_cache(tmp_path):
    source=tmp_path/'source';cache=tmp_path/'cache';source.mkdir();cache.mkdir()
    def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')
    geo=dict(static=np.zeros((2,7),dtype='float32'),coast=np.array([0,10],dtype='float32'),
        dev_static=np.zeros((1,7),dtype='float32'),dev_coast=np.array([20],dtype='float32'))
    np.savez(source/'geometry.npz',**geo)
    write(source/'contract.json',dict(files={'geometry.npz':sha(source/'geometry.npz')}))
    write(cache/'protocol.json',dict(config=dict(data=str(source),calendar_day=0),data_sha256=sha(source/'contract.json')))
    np.save(cache/'mean.npy',np.zeros((1,12,2),dtype='float32'))
    write(cache/'geometry.json',dict(train_static=static(geo).tolist(),dev_static=static(geo,'dev_').tolist()))
    write(cache/'status.json',dict(status='COMPLETE',completed=1))
    write(cache/'manifest.json',dict(files={'mean.npy':sha(cache/'mean.npy')},protocol_sha256=sha(cache/'protocol.json')))
    experiment.CachedData(cache,device='cpu')
    return cache


def test_cache_role_protocol_cannot_change_after_manifest_lock(locked_cache):
    path=locked_cache/'protocol.json';value=json.loads(path.read_text());value['config']['calendar_day']=8
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='protocol'):experiment.CachedData(locked_cache,device='cpu')


def test_query_geometry_cannot_change_without_a_source_contract_change(locked_cache):
    path=locked_cache/'geometry.json';value=json.loads(path.read_text());value['train_static'][0][0]=123.
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='geometry'):experiment.CachedData(locked_cache,device='cpu')
