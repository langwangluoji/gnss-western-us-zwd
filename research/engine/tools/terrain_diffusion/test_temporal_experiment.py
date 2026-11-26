import numpy as np
import json
import hashlib
import pytest


def test_station_macro_differs_from_micro_for_unequal_station_error():
    from .temporal_experiment import score
    truth = np.zeros((2, 6, 2))
    prediction = np.zeros_like(truth)
    prediction[:, :, 1] = 20
    metrics = score(prediction, truth)
    assert metrics['macro_rmse_mm'] == 10
    np.testing.assert_allclose(metrics['micro_rmse_mm'], np.sqrt(200))
    assert metrics['increment_macro_rmse_mm'] == 0


def test_score_masks_missing_targets():
    from .temporal_experiment import score
    truth = np.zeros((2, 6, 2))
    truth[:, :, 0] = np.nan
    prediction = np.full_like(truth, 3)
    metrics = score(prediction, truth)
    assert metrics['macro_rmse_mm'] == 3
    assert metrics['station_count'] == 1


def test_changed_data_file_rejected_before_training(tmp_path):
    from .temporal_experiment import Events
    for name in ('hours','dev_hours'):
        np.save(tmp_path/f'{name}.npy',np.arange(100))
    for name in ('values','dev_values'):
        np.save(tmp_path/f'{name}.npy',np.ones((100,2,3),dtype='float32'))
    np.savez(tmp_path/'geometry.npz',xy=np.array([[0.,0.],[1.,1.]]))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    (tmp_path/'contract.json').write_text(json.dumps(dict(files=hashes)),encoding='utf-8')
    np.save(tmp_path/'values.npy',np.zeros((100,2,3),dtype='float32'))
    with pytest.raises(ValueError,match='hash'):
        Events(tmp_path)
