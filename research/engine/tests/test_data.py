import numpy as np
import pytest
import torch


def test_role_views_and_statistics_do_not_consume_holdout_labels(legacy_fixture, tmp_path):
    from gnss_field.data import pack_legacy, PackedStore, ContextBuilder
    a = tmp_path/'pack1'
    pack_legacy(legacy_fixture, a, registered=False)
    values = np.load(legacy_fixture/'observations.npy')
    values[:, 8:, :] = 9e5
    np.save(legacy_fixture/'observations.npy', values)
    b = tmp_path/'pack2'
    pack_legacy(legacy_fixture, b, registered=False)
    s1, s2 = PackedStore(a), PackedStore(b)
    assert s1.stats == s2.stats
    assert s1.manifest['training_hash'] == s2.manifest['training_hash']
    hours = s1.hours[20:22]
    c1, c2 = ContextBuilder(s1).build_context(hours), ContextBuilder(s2).build_context(hours)
    for key in ['stations', 'padding', 'coarse', 'maps', 'baseline']:
        torch.testing.assert_close(getattr(c1, key), getattr(c2, key))
    with pytest.raises(PermissionError):
        s1.targets('C', hours)
    with pytest.raises(PermissionError):
        s1.targets('D', hours)
    with pytest.raises(PermissionError):
        PackedStore(a, purpose='retrospective', retrospective_lock={})


def test_current_outage_and_persistent_outage_rebuild_frames(legacy_fixture, tmp_path):
    from gnss_field.data import pack_legacy, PackedStore, ContextBuilder, InputScenario
    p = tmp_path/'packed'
    pack_legacy(legacy_fixture, p, registered=False)
    store = PackedStore(p)
    build = ContextBuilder(store)
    h = store.hours[24:25]
    current = build.build_context(h, InputScenario(retained=(), persistent=False))
    persistent = build.build_context(h, InputScenario(retained=(), persistent=True))
    assert current.padding.all() and persistent.padding.all()
    assert current.coarse[:, 0, :6].count_nonzero() == 0
    assert persistent.coarse[:, :, :6].count_nonzero() == 0
    assert current.scenario_hash != persistent.scenario_hash
    assert torch.isfinite(persistent.baseline).all()
    one = build.build_context(h, InputScenario(retained=(0,), persistent=True))
    torch.testing.assert_close(one.baseline, torch.full_like(one.baseline, one.baseline.flatten()[0]))
    assert not one.maps[:, 3].bool().any()


def test_pack_is_idempotent_and_detects_source_change(legacy_fixture, tmp_path):
    from gnss_field.data import pack_legacy
    p = tmp_path/'packed'
    first = pack_legacy(legacy_fixture, p, registered=False)
    assert first == pack_legacy(legacy_fixture, p, registered=False)
    a = np.load(legacy_fixture/'observations.npy')
    a[0, 0, 0] += 1
    np.save(legacy_fixture/'observations.npy', a)
    with pytest.raises(ValueError):
        pack_legacy(legacy_fixture, p, registered=False)
