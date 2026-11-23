import numpy as np


def test_prediction_scores_missing_truth_with_geometric_support(legacy_fixture, tmp_path):
    from gnss_field.data import PackedStore, pack_legacy
    from gnss_field.training.prediction import evaluate_predictor

    observations = np.load(legacy_fixture/'observations.npy')
    observations[30, 8, 0] = np.nan
    np.save(legacy_fixture/'observations.npy', observations)
    pack_legacy(legacy_fixture, tmp_path/'packed', registered=False)
    store = PackedStore(tmp_path/'packed', purpose='audit')
    hours = np.asarray(store.hours)[30:32]

    def predictor(hours, context, ids, members):
        prediction = np.full((len(hours), len(ids)), 100.)
        return dict(mean=prediction, coarsened=prediction)

    result = evaluate_predictor(store, hours, predictor, tmp_path/'evaluation')
    assert result['hours'] == 2
    # Cached geometric support must also work on a fresh scoring pass.
    (tmp_path/'evaluation/metrics.json').unlink()
    repeated = evaluate_predictor(store, hours, predictor, tmp_path/'evaluation')
    assert repeated['values'] == result['values']
