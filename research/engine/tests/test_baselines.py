import numpy as np
import torch


def test_idw_baseline_uses_current_input_roles_only(legacy_fixture, tmp_path):
    from gnss_field.data import pack_legacy, PackedStore, ContextBuilder, InputScenario
    from gnss_field.training.baselines import get_baseline_predictor
    pack_legacy(legacy_fixture, tmp_path/'pack', registered=False)
    store = PackedStore(tmp_path/'pack')
    builder = ContextBuilder(store)
    h = store.hours[24:25]
    ctx = builder.build_context(h, InputScenario(retained=(0,), persistent=True))
    predictor = get_baseline_predictor('A_IDW', store, None)
    row = predictor(h, ctx, store.ids('C'), 64)
    expected = store.context_values(h)[0, 0, 0]*np.exp(-store.stations.height.to_numpy()[store.ids('C')]/3000)
    np.testing.assert_allclose(row['mean'][0], expected, atol=1e-4)
    assert row.get('samples') is None


def test_ordinary_kriging_constant_preservation():
    from gnss_field.training.baselines import ordinary_kriging
    xy = np.array([[0., 0.], [10000., 0.], [0., 10000.]])
    q = np.array([[5000., 5000.], [200000., 200000.]])
    pred = ordinary_kriging(xy, np.array([100., 100., 100.]), q,
                             {'sill': 50., 'range_km': 100., 'nugget': .1})
    np.testing.assert_allclose(pred, 100., atol=1e-10)
