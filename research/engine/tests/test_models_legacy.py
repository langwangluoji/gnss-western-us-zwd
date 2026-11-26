"""Parity uses historical sources only in these tests; production stays portable."""
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
LEGACY = REPOSITORY / 'artifacts/legacy_layout/research_pipeline/studies/us_dense_zwd_20261105_v1/03_code'
if not (LEGACY / 'field_generation_v1/model.py').is_file():
    LEGACY = REPOSITORY / 'archive/legacy/research_pipeline/studies/us_dense_zwd_20261105_v1/03_code'
torch.set_num_threads(2)


def standalone(name):
    try:
        return importlib.import_module('gnss_field.models.' + name)
    except ModuleNotFoundError as exc:
        pytest.fail(f'Standalone {name} model is missing: {exc}')


def legacy():
    if not (LEGACY / 'field_generation_v1' / 'model.py').is_file():
        pytest.skip('Historical sources are available only in the development tree')
    sys.path.insert(0, str(LEGACY))
    try:
        return importlib.import_module('field_generation_v1.model')
    finally:
        sys.path.remove(str(LEGACY))


def patch(batch=2, stations=7):
    return dict(stations=torch.randn(batch, stations, 17),
                padding=torch.zeros(batch, stations, dtype=torch.bool),
                coarse=torch.randn(batch, 4, 15, 6, 6),
                maps=torch.randn(batch, 6, 24, 24),
                baseline=torch.randn(batch, 1, 6, 6),
                baseline_condition=torch.randn(batch, 1, 24, 24),
                mu_condition=torch.randn(batch, 1, 24, 24),
                domain=torch.ones(batch, 1, 24, 24))


def test_m0_exact_checkpoint_keys_and_nontrivial_output_parity():
    new = standalone('mean').MeanField()
    old = legacy().MeanField()
    torch.manual_seed(19)
    state = {name: torch.randn_like(value) * .03 for name, value in old.state_dict().items()}
    old.load_state_dict(state)
    assert set(new.state_dict()) == set(state)
    new.load_state_dict(state, strict=True)
    old.eval(); new.eval()
    for count, empty in [(7, False), (7, True), (0, True)]:
        b = patch(stations=count)
        b['padding'][:] = empty
        with torch.no_grad():
            expected, _ = old(b)
            actual, extra = new(b)
        assert actual.shape == (2, 1, 24, 24) and extra is None
        torch.testing.assert_close(actual, expected, rtol=0, atol=.001)


def test_m2_parameters_features_and_outputs_match_actual_legacy():
    new = standalone('pixel').ResidualField()
    old = legacy().ResidualField()
    torch.manual_seed(25)
    state = {name: torch.randn_like(value) * .03 for name, value in old.state_dict().items()}
    old.load_state_dict(state); new.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in new.parameters()) == sum(p.numel() for p in old.parameters())
    for empty in (False, True):
        b = patch(); b['padding'][:] = empty
        with torch.no_grad():
            expected, old_features = old(b)
            actual, features = new(b)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(features, old_features, rtol=0, atol=0)
        assert features.shape == (2, 32, 24, 24)


@pytest.mark.parametrize('faithful', [False, True])
def test_faithful_loss_and_variance_gradient_parity(faithful):
    pixel = standalone('pixel')
    old = legacy().ResidualField(); new = pixel.ResidualField()
    with torch.no_grad():
        old.variance_head[-1].weight.fill_(.07)
    new.load_state_dict(old.state_dict())
    feature = torch.randn(4, 32, requires_grad=True)
    other = feature.detach().clone().requires_grad_(True)
    geometry = torch.randn(4, 3); target = torch.randn(4)
    std = new.standard_deviation(feature, geometry, 20., faithful)
    expected_std = old.standard_deviation(other, geometry, 20., faithful)
    torch.testing.assert_close(std, expected_std, rtol=0, atol=0)
    mse, nll = pixel.gaussian_losses(feature[:, 0], target, std.square(), 20., faithful)
    old_mse, old_nll = legacy().gaussian_losses(other[:, 0], target, expected_std.square(), 20., faithful)
    actual = torch.autograd.grad(mse + nll, feature)[0]
    expected = torch.autograd.grad(old_mse + old_nll, other)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if faithful:
        hand_mean_gradient = (feature[:, 0].detach() - target) / (400 * 4)
        torch.testing.assert_close(actual[:, 0], hand_mean_gradient)
        assert torch.count_nonzero(actual[:, 1:]) == 0
    else:
        assert torch.count_nonzero(actual[:, 1:]) > 0


def test_models_import_and_run_without_historical_modules():
    standalone('mean'); standalone('pixel')
    script = '''
import importlib.abc, sys, torch
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'architecture_search', 'regional_temporal_mae', 'mae_gan_cvae',
                'cvae_pilot', 'moment_superresolution', 'station_mae', 'us_gan', 'field_generation_v1',
                'residual_edm_v1', 'train_us_gan', 'feature_learning', 'model', 'h5py', 'pandas'}:
            raise RuntimeError('Historical or data import: ' + fullname)
sys.meta_path.insert(0, BlockLegacy())
from gnss_field.models.mean import MeanField
from gnss_field.models.pixel import ResidualField
from gnss_field.models.condition import ConditionEncoder
from gnss_field.models.basis import SpatialBasis
from gnss_field.models.heads import DeterministicHead, GaussianHead, GANGenerator, ConditionalCritic
from gnss_field.models.diffusion import EDMHead
from gnss_field.contracts import ContextBatch
torch.set_num_threads(2)
b = dict(stations=torch.randn(1,2,17), padding=torch.zeros(1,2,dtype=torch.bool),
    coarse=torch.randn(1,4,15,6,6), maps=torch.randn(1,6,24,24), baseline=torch.zeros(1,1,6,6),
    baseline_condition=torch.zeros(1,1,24,24), mu_condition=torch.zeros(1,1,24,24),
    domain=torch.ones(1,1,24,24))
assert MeanField()(b)[0].shape == (1,1,24,24)
assert ResidualField()(b)[1].shape == (1,32,24,24)
c = ContextBatch(hours=torch.tensor([1]), stations=b['stations'], padding=b['padding'],
    coarse=b['coarse'], maps=b['maps'], baseline=b['baseline_condition'])
h = ConditionEncoder()(c)
phi = SpatialBasis(rank=4)(torch.randn(10,3), torch.ones(10,dtype=torch.bool))
assert phi.shape == (10,4)
assert DeterministicHead(rank=4)(h).shape == (1,4)
assert GaussianHead(rank=4)(h)[0].shape == (1,4)
generated = GANGenerator(rank=4)(torch.randn(1,4),h)
assert ConditionalCritic(rank=4)(generated,h).shape == (1,)
assert EDMHead(rank=4).sample(h,[1],2,44).shape == (1,2,4)
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
