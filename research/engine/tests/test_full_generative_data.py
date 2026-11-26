"""Input-only compact fields, support, role gates and resumable source identity."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gnss_field.artifacts import sha256
from gnss_field.data import PackedStore, pack_legacy


class ConstantBackground:
    identity = 'synthetic-frozen-background-120mm-v1'

    def __call__(self, hours, frames):
        return np.full((len(hours), 1, 24, 24), 120., dtype='f4')


@pytest.fixture
def field_data(legacy_fixture, tmp_path):
    from gnss_field.generative.data import FieldDataConfig, FullFieldData
    # Enclose every positive-weight target stencil within the A convex hull.
    import pandas as pd
    stations = pd.read_csv(legacy_fixture/'station_roles.csv')
    stations.loc[:3, ['x', 'y']] = [[175000, 175000], [275000, 175000], [175000, 275000], [275000, 275000]]
    stations.to_csv(legacy_fixture/'station_roles.csv', index=False)
    root = tmp_path/'packed'
    pack_legacy(legacy_fixture, root, registered=False)
    config = FieldDataConfig(packed_root=str(root), cache_root=str(tmp_path/'compact'))
    data = FullFieldData(config, background_provider=ConstantBackground())
    return data


def test_schema_four_exact_frames_and_pure_bilinear_height_separation(field_data):
    data = field_data
    hour = int(data.store.hours[30])
    out = data.batch([hour])
    assert out['grid_condition'].shape == (1, 37, 24, 24)
    assert out['station_tokens'].shape == (1, 4, 31)
    assert out['station_mask'].all()
    assert out['support'].any()
    assert out['target_mask'].all()
    mean = data.input_mean[0]
    scale = data.input_std[0]
    # The source fixture is 100+0.1*hour_index+0.3*station_index.
    expected = torch.tensor([(103.-.1*offset-mean)/scale for offset in (0, 6, 12, 18)])
    torch.testing.assert_close(out['station_tokens'][0, 0, [0, 5, 10, 15]], expected.float())
    torch.testing.assert_close(out['station_tokens'][0, :, 20:24], torch.ones(4, 4))
    torch.testing.assert_close(out['query_weights'].sum(-1), torch.ones(4))
    # All four B fixture locations are exact pixel centres, despite positive heights.
    torch.testing.assert_close(out['query_weights'][:, 0], torch.ones(4))
    assert (out['target_geometry'][:, 2] > 0).all()
    assert out['query_indices'][0].tolist() == [200, 201, 224, 225]
    # Daily cycle comes before annual cycle in the new schema (06:00 here).
    torch.testing.assert_close(out['station_tokens'][0, 0, 27:29], torch.tensor([1., 0.]), atol=1e-6, rtol=0)
    assert data.metadata['coarse_shape'] == [6, 6]
    assert data.metadata['height_datum'] == 'EGM96'


def test_condition_does_not_read_target_values_or_change_after_label_perturbation(field_data):
    data = field_data
    hour = int(data.store.hours[30])
    before = data.condition_batch([hour], 'cpu')
    original = data.store.targets

    def changed(role, hours):
        return original(role, hours) + 9999

    data.store.targets = changed
    after = data.condition_batch([hour], 'cpu')
    for key in before:
        torch.testing.assert_close(before[key], after[key])
    assert not any(k.startswith('target') for k in before)
    labels = data.batch([hour])
    assert labels['target'].min() > 9000

    def forbidden(*args):
        raise AssertionError('Condition path attempted target access')

    data.store.targets = forbidden
    data.condition_batch([hour], 'cpu')


@pytest.mark.parametrize('keep', [0, 1, 4])
def test_zero_single_and_full_missing_are_finite_and_joint_valid(field_data, keep):
    data = field_data
    original = data.store.context_values

    def outage(hours):
        values = original(hours).copy()
        # A missing *single* variable invalidates the complete five-variable frame.
        values[:, keep:, 3] = np.nan
        return values

    data.store.context_values = outage
    out = data.batch([int(data.store.hours[30])])
    assert out['station_mask'].sum() == keep
    assert torch.isfinite(out['grid_condition']).all()
    assert torch.isfinite(out['station_tokens']).all()
    assert out['station_tokens'][0, keep:, :24].count_nonzero() == 0
    if keep < 3:
        assert not out['support'].any()
        assert not out['target_mask'].any()
    if keep == 0:
        assert out['grid_condition'][:, :24].count_nonzero() == 0
    if keep == 1:
        # One valid A supplies a spatially constant IDW coarse field.
        assert out['grid_condition'][0, 0].std() < 1e-6


def test_full_hour_eligibility_19h_boundary_missing_labels_and_role_gates(field_data):
    data = field_data
    original = data.store.targets
    bad = int(data.store.hours[30])

    def labels(role, hours):
        values = original(role, hours).copy()
        values[np.asarray(hours) == bad, 0] = np.nan
        return values

    data.store.targets = labels
    hours = data.train_hours
    assert len(hours) == 77  # 96 hours minus first 18, minus one B<4 hour.
    assert hours[0] == data.store.hours[18]
    assert bad not in hours
    audit = data.audit('B')
    assert audit['raw_hours'] == 96
    assert audit['reason_counts']['history_unavailable'] == 18
    assert audit['reason_counts']['fewer_than_four_supported_labels'] == 1
    with pytest.raises(ValueError, match='history'):
        data.condition_batch(data.store.hours[17:18], 'cpu')
    for role in ('C', 'D'):
        with pytest.raises(PermissionError):
            data.batch(data.store.hours[30:31], role=role)


def test_monthly_compact_cache_is_atomic_resumable_and_reads_are_identical(field_data):
    data = field_data
    hours = data.store.hours[24:27]
    expected = data.batch(hours)
    result = data.prepare(hours)
    assert result['prepared_hours'] == 3
    assert result['condition_shards'] == 1
    files = list(Path(data.config.cache_root).rglob('*.npy'))
    assert len(files) == 10  # Eight condition arrays and separately gated hours/B labels.
    cond = next(p for p in files if p.name == 'coarse.npy')
    coarse = np.load(cond, mmap_mode='r')
    assert isinstance(coarse, np.memmap)
    assert coarse.shape == (3, 20, 6, 6)
    assert not list(cond.parent.glob('*target*'))
    assert not any(np.load(p, mmap_mode='r').shape == (3, 37, 24, 24) for p in cond.parent.glob('*.npy'))
    timestamps = {p: p.stat().st_mtime_ns for p in files}
    assert data.prepare(hours)['prepared_hours'] == 3
    assert timestamps == {p: p.stat().st_mtime_ns for p in files}
    assert not list(Path(data.config.cache_root).rglob('*.partial'))
    data.background_provider = lambda *args: (_ for _ in ()).throw(AssertionError('Cache miss'))
    actual = data.batch(hours)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key])
    # A damaged committed shard must fail closed, not silently regenerate.
    with cond.open('ab') as f:
        f.write(b'corrupt')
    data._clear_cache()
    with pytest.raises(ValueError, match='checksum'):
        data.condition_batch(hours, 'cpu')


def test_cache_preparation_extends_partial_month_without_losing_old_rows(field_data):
    data = field_data
    h = data.store.hours[24:28]
    data.prepare(h[:2])
    data.prepare(h[2:])
    pointer = next(Path(data.config.cache_root).rglob('conditions/2016-01.json'))
    record = json.loads(pointer.read_text())
    assert np.load(pointer.parent/record['directory']/'hour_ids.npy').tolist() == h.tolist()
    assert data.prepare(h)['prepared_hours'] == 4


def test_verified_background_assets_reject_mutation_and_missing_provider(field_data, tmp_path):
    from gnss_field.generative.data import VerifiedBackgroundProvider, FullFieldData
    asset = tmp_path/'frozen.bin'
    asset.write_bytes(b'frozen geometry coefficients')
    assets = {str(asset): sha256(asset)}
    provider = VerifiedBackgroundProvider(ConstantBackground(), assets)
    data = FullFieldData(field_data.config, background_provider=provider)
    assert data.condition_batch(data.store.hours[30:31], 'cpu')['background'].mean() == 120
    asset.write_bytes(b'changed geometry coefficients')
    with pytest.raises(ValueError, match='asset'):
        VerifiedBackgroundProvider(ConstantBackground(), assets)
    with pytest.raises(ValueError, match='background'):
        FullFieldData(field_data.config)


def test_training_and_development_are_distinct_permission_views(field_data):
    from gnss_field.generative.data import FullFieldData
    dev = FullFieldData(field_data.config, purpose='development', background_provider=ConstantBackground())
    assert len(dev.development_hours) == 0  # This fixture contains 2016 only.
    with pytest.raises(PermissionError):
        dev.batch(dev.store.hours[30:31], role='B')


def test_interrupted_month_commit_preserves_old_pointer_and_can_resume(field_data, monkeypatch):
    import gnss_field.generative.data as module
    data = field_data
    hours = data.store.hours[24:28]
    data.prepare(hours[:2])
    real_write = module.write_json

    def interrupt(path, value):
        if Path(path).parent.name == 'conditions':
            raise OSError('Interrupted before atomic month pointer publication')
        return real_write(path, value)

    monkeypatch.setattr(module, 'write_json', interrupt)
    with pytest.raises(OSError, match='Interrupted'):
        data.prepare(hours[2:])
    monkeypatch.setattr(module, 'write_json', real_write)
    data._clear_cache()
    assert data.batch(hours[:2])['hour_ids'].tolist() == hours[:2].tolist()
    assert data.prepare(hours)['prepared_hours'] == 4
    assert data.batch(hours)['hour_ids'].tolist() == hours.tolist()


def test_normalization_uses_all_training_A_and_ignores_old_frozen_statistics(field_data):
    from copy import copy
    from gnss_field.generative.data import FullFieldData
    store = copy(field_data.store)
    base_hours = np.asarray(store.hours)
    delta = int(np.datetime64('2018-01-01', 'h').astype('i8'))-int(base_hours[0])
    store.hours = np.r_[base_hours, base_hours+delta]
    original = store.context_values

    def values(hours):
        hours = np.asarray(hours)
        late = hours >= base_hours[0]+delta
        raw = original(hours-late*delta).copy()
        raw[..., 0] += late[:, None]*100
        return raw

    store.context_values = values
    store.stats = dict(store.stats, input_mean=[-9999]*5, input_std=[9999]*5)
    data = FullFieldData(field_data.config, store=store, background_provider=ConstantBackground())
    assert data.input_mean[0] == pytest.approx(155.2, abs=1e-5)
    assert data.metadata['statistics']['station_hours'] == 768
    assert data.metadata['statistics']['source'] == 'joint_valid_A_only'


def test_current_missing_support_does_not_hide_available_historical_tokens(field_data):
    data = field_data
    hour = int(data.store.hours[30])
    original = data.store.context_values

    def values(hours):
        raw = original(hours).copy()
        raw[np.asarray(hours) == hour] = np.nan
        return raw

    data.store.context_values = values
    out = data.batch([hour])
    assert out['station_mask'].all()  # Attention may use the past three valid frames.
    assert not out['support'].any()  # Physical support always uses the current A set.
    assert not out['target_mask'].any()
    assert out['station_tokens'][..., :5].count_nonzero() == 0
    assert out['station_tokens'][..., 5:20].count_nonzero() > 0


def test_month_mmaps_survive_random_three_month_access_and_prepare_bounds_provider(legacy_fixture, tmp_path, monkeypatch):
    from gnss_field.generative.data import FieldDataConfig, FullFieldData
    values = np.tile(np.load(legacy_fixture/'observations.npy'), (17, 1, 1))
    np.save(legacy_fixture/'observations.npy', values)
    audit_path = legacy_fixture/'audit.json'
    audit = json.loads(audit_path.read_text())
    audit['stop'] = audit['start']+len(values)
    audit_path.write_text(json.dumps(audit))
    packed = tmp_path/'packed_three_months'
    pack_legacy(legacy_fixture, packed, registered=False)

    class BoundedBackground(ConstantBackground):
        def __call__(self, hours, frames):
            assert len(hours) <= 2, 'Background preparation exceeded bounded microbatch'
            return super().__call__(hours, frames)

    data = FullFieldData(FieldDataConfig(packed_root=str(packed), cache_root=str(tmp_path/'months'),
                        prepare_batch_hours=2, mapped_month_capacity=6), background_provider=BoundedBackground())
    h = data.store.hours[[24, 25, 26, 27, 28, 744+24, 744+25, 1440+24, 1440+25]]
    data.prepare(h)
    data._clear_cache()
    real_load = np.load
    opened = []

    def load(path, *args, **kwargs):
        if Path(path).name == 'coarse.npy':
            opened.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', load)
    for hour in (h[0], h[5], h[7], h[0]):
        out = data.condition_batch([hour], 'cpu')
        assert torch.isfinite(out['grid_condition']).all()
    assert len(opened) == 3  # Revisit A after B,C performs no second A mmap/open.


@pytest.mark.parametrize('role,purpose', [('B', 'train'), ('C', 'development')])
def test_qualified_audit_reloads_exact_hours_and_reference_without_observation_reads(field_data, role, purpose):
    from gnss_field.generative.data import FullFieldData
    data = FullFieldData(field_data.config, purpose=purpose, background_provider=ConstantBackground())
    expected = data.audit(role)
    again = FullFieldData(field_data.config, purpose=purpose, background_provider=ConstantBackground())

    def forbidden(*args):
        raise AssertionError('Persisted eligibility audit reread observations')

    again.store.context_values = forbidden
    again.store.targets = forbidden
    actual = again.audit(role)
    assert actual == expected
    if role == 'B':
        assert actual['qualified_hours'] == 78
        assert actual['reference_count'] == 4.0
        assert again.train_hours.tolist() == field_data.store.hours[18:].tolist()
    assert list(Path(data.config.cache_root).rglob('audits/*.json'))
    assert not list(Path(data.config.cache_root).rglob('*.partial'))


def test_changed_label_release_recomputes_audit_and_does_not_reuse_old_memory_report(field_data):
    from copy import deepcopy
    from gnss_field.artifacts import fingerprint, write_json
    data = field_data
    original = data.audit('B')
    source = data.store.root/'B/2016-01.npy'
    data.store._shard.cache_clear()
    values = np.load(source)
    values[30, 0] = np.nan
    np.save(source, values)
    data.store.manifest = deepcopy(data.store.manifest)
    data.store.manifest['files']['B/2016-01.npy']['sha256'] = sha256(source)
    binding = dict(data.store.manifest)
    binding.pop('fingerprint')
    data.store.manifest['fingerprint'] = fingerprint(binding)
    write_json(data.store.root/'manifest.json', data.store.manifest)
    actual = data.audit('B')
    assert actual['labels_identity'] != original['labels_identity']
    assert actual['qualified_hours'] == 77
    assert int(data.store.hours[30]) not in actual['eligible_hours']
    assert original['qualified_hours'] == 78


def test_tampered_audit_reference_fails_integrity_check_before_training(field_data):
    from gnss_field.generative.data import FullFieldData
    data = field_data
    data.audit('B')
    path = next(Path(data.config.cache_root).rglob('audits/*.json'))
    contents = json.loads(path.read_text())
    contents['report']['reference_count'] = 9999
    path.write_text(json.dumps(contents))
    again = FullFieldData(field_data.config, background_provider=ConstantBackground())
    with pytest.raises(ValueError, match='audit cache'):
        again.audit('B')


def test_interrupted_audit_commit_can_retry_and_publish_complete_report(field_data, monkeypatch):
    import gnss_field.generative.data as module
    real_write = module.write_json

    def interrupt(path, value):
        if Path(path).parent.name == 'audits':
            raise OSError('Interrupted eligibility commit')
        return real_write(path, value)

    monkeypatch.setattr(module, 'write_json', interrupt)
    with pytest.raises(OSError, match='Interrupted'):
        field_data.audit('B')
    assert not list(Path(field_data.config.cache_root).rglob('audits/*.json'))
    monkeypatch.setattr(module, 'write_json', real_write)
    expected = field_data.audit('B')
    path = next(Path(field_data.config.cache_root).rglob('audits/*.json'))
    assert json.loads(path.read_text())['report'] == expected
