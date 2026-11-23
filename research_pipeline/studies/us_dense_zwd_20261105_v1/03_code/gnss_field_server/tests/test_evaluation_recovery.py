import importlib.util
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gnss_field.artifacts import immutable_json
from gnss_field.contracts import ScienceConfig
from gnss_field.models.mean import MeanField
from gnss_field.training.engine import StageSpec, TrainEngine, atomic_torch


def recovery_module():
    path = Path(__file__).parents[1]/'tools/recover_m0_evaluation.py'
    spec = importlib.util.spec_from_file_location('recovery_tool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_preserves_next_update_and_rejects_science_change(tmp_path):
    migrate = recovery_module().migrate_checkpoint
    science = replace(ScienceConfig(), evaluate_every=500, log_every=1)
    hours = np.arange(8)
    spec = StageSpec('mean_base', 5000, 5000, 8, 2, 42, {'release_sha256': 'old'})
    engine = TrainEngine(MeanField(), tmp_path/'original', hours, spec, science)
    engine.step = 200
    engine.pending_evaluation = True
    state = engine.state()
    contract = dict(signature=engine.signature, spec=asdict(spec), science=asdict(science))
    migrated = migrate(state, contract, science, hours, 'old', 'new')
    for key in state.keys() - {'signature', 'spec'}:
        assert migrated[key] is state[key]
    assert state['spec']['dependencies']['release_sha256'] == 'old'
    with pytest.raises(ValueError, match='Scientific'):
        migrate(state, contract, replace(science, learning_rate=.002), hours, 'old', 'new')
    with pytest.raises(ValueError, match='signature'):
        migrate(state, contract, science, hours+1, 'old', 'new')
    for folder, saved, stage in (
        ('reference', state, spec),
        ('recovered', migrated, replace(spec, dependencies={'release_sha256': 'new'})),
    ):
        directory = tmp_path/folder
        atomic_torch(directory/'last.pt', saved)
        immutable_json(directory/'contract.json', dict(signature=saved['signature'],
                       spec=asdict(stage), science=asdict(science)))
        resumed = TrainEngine(MeanField(), directory, hours, stage, science)
        resumed.pending_evaluation = False
        resumed.run(lambda batch, model, step: {'loss': next(model.parameters()).square().mean()},
                    stop_after=201)
    expected = torch.load(tmp_path/'reference/last.pt', weights_only=False)
    actual = torch.load(tmp_path/'recovered/last.pt', weights_only=False)
    assert actual['step'] == expected['step'] == 201
    assert actual['cursor']['position'] == expected['cursor']['position']
    for key in ('model', 'ema'):
        assert all(torch.equal(actual[key][name], tensor) for name, tensor in expected[key].items())
