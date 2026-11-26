import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from residual_edm_v1.evaluation import (
    evaluate, paired_block_bootstrap, point_metrics, probability_metrics,
    sample_stations, station_support,
)


class FakeData:
    hours = np.arange(4, dtype=np.int64)
    scale = 2.
    teacher_complete = np.array([True, False, True, False])
    manifest = {'event_indices': [0]}

    def batch(self, indices, targets=False):
        assert not targets
        count = len(indices)
        return dict(mu_mm=torch.full((count, 1, 24, 24), 20.),
                    domain=torch.ones(1, 1, 24, 24, dtype=torch.bool),
                    context_support=torch.ones(count, 1, 24, 24))

    def evaluation(self, role, years):
        assert role == 'validation' and tuple(years) == (2019,)
        return dict(truth=np.full((4, 2), 10.), xy=np.array([[0., 0.], [.2, .1]]),
                    factor=np.array([.5, .5]), names=['A', 'B'])


class EvaluationTests(unittest.TestCase):
    def test_grid_sampling_height_and_boundary_support(self):
        field = torch.full((2, 1, 24, 24), 20.)
        xy = torch.tensor([[0., 0.], [1., 0.]])
        np.testing.assert_allclose(sample_stations(field, xy, torch.tensor([.5, .5]))[:, 0], 10.)
        support = station_support(torch.ones_like(field), xy)
        self.assertTrue(support[:, 0].all())
        self.assertFalse(support[:, 1].any())

    def test_macro_rmse_recomputed_per_station(self):
        truth = np.zeros((2, 2))
        pred = np.array([[1., 3.], [1., 0.]])
        valid = np.array([[True, True], [True, False]])
        result = point_metrics(pred, truth, valid)
        self.assertAlmostEqual(result['station_macro_rmse_mm'], 2.)
        self.assertEqual(result['station_records'], 3)
        pred[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            point_metrics(pred, truth, valid)

    def test_probability_scores_known_ensemble(self):
        samples = np.array([[[0.], [2.]]])
        result = probability_metrics(samples, np.array([[1.]]), np.array([[True]]))
        self.assertAlmostEqual(result['crps_mm'], .5)
        self.assertAlmostEqual(result['energy_score_mm'], .5)
        self.assertEqual(result['coverage_80'], 1.)

    def test_mean_evaluation_does_not_request_targets(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate(None, FakeData(), np.arange(4), output=Path(tmp), probability=True)
            self.assertEqual(result['station_macro_rmse_mm'], 0.)
            self.assertEqual(result['station_records'], 8)
            self.assertTrue((Path(tmp) / 'evaluation.h5').exists())
            self.assertEqual(result['teacher_complete_group']['station_records'], 4)
            with h5py.File(Path(tmp) / 'evaluation.h5', 'r') as handle:
                self.assertEqual(handle['event_samples/z0_mm'].shape, (1, 1, 24, 24))
                self.assertEqual(handle['event_samples/hour_id'][0], 0)

    def test_bootstrap_shared_time_blocks_and_difference(self):
        errors = np.ones((400, 2))
        result = paired_block_bootstrap(errors, errors * 2, np.ones_like(errors, bool),
                                        np.arange(400), repeats=50, seed=11)
        self.assertAlmostEqual(result['delta_macro_rmse_mm'], -1.)
        np.testing.assert_allclose(result['ci95_mm'], [-1., -1.])

    def test_gaussian_uses_analytic_point_mean(self):
        class Gaussian(torch.nn.Module):
            arm = 'gaussian'

            def forward(self, batch):
                return torch.ones_like(batch['mu_mm']), torch.zeros_like(batch['mu_mm'])

            def encode_condition(self, batch):
                return batch

            def sample(self, condition, hours, members, seed):
                return torch.full((len(hours), members, 1, 24, 24), 100.)

        result = evaluate(Gaussian(), FakeData(), np.arange(4), members=2, probability=True)
        self.assertAlmostEqual(result['station_macro_rmse_mm'], 1.)
        self.assertGreater(result['probability']['crps_mm'], 90.)

    def test_nonfinite_prediction_is_not_dropped(self):
        class BrokenData(FakeData):
            def batch(self, indices, targets=False):
                result = super().batch(indices, targets)
                result['mu_mm'][:] = float('nan')
                return result

        with self.assertRaisesRegex(ValueError, 'Non-finite prediction'):
            evaluate(None, BrokenData(), np.arange(4))


if __name__ == '__main__':
    unittest.main()
