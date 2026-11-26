"""Read-only ensemble audit and a scalar Monte Carlo check of mean-MSE bias."""
import json
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'results/20261123_enhanced_encoder_generative_v03'
OUT = PROJECT/'results/20261123_postrun_design_v01'


def macro_rmse(prediction, target, mask):
    counts = mask.sum(0)
    squared = np.where(mask, prediction-target, 0.)**2
    return float(np.sqrt(squared.sum(0)[counts > 0]/counts[counts > 0]).mean())


def main():
    OUT.mkdir(exist_ok=True)
    baseline = np.load(PROJECT/'results/20261123_enhanced_encoder_generative_v01/encoder_seed42/C2019_development_predictions.npz')
    results = {}
    for family in ('vae', 'gan', 'diffusion'):
        results[family] = {}
        for product in ('point', 'joint'):
            archive = np.load(SOURCE/family/f'{product}_C2019_development.npz')
            x = archive['samples'].astype('f8')
            y, mask = archive['target'], archive['mask']
            for key in ('station_ids', 'target', 'mask'):
                np.testing.assert_array_equal(archive[key], baseline[key])
            np.testing.assert_array_equal(archive['hours'], baseline['hour_ids'])
            k = x.shape[1]
            record = dict(prefix_macro_rmse={str(n): macro_rmse(x[:, :n].mean(1), y, mask) for n in (4, 8, 16, 32)})
            sample_variance = x.var(1, ddof=1)
            mse = np.where(mask, (x.mean(1)-y)**2, 0.).sum()/mask.sum()
            correction = sample_variance[mask].mean()/k
            record.update(micro_mse=float(mse), estimated_member_mean_mse_penalty=float(correction),
                          corrected_micro_mse=float(mse-correction),
                          mean_shift_from_deterministic_initializer_mm=float((x.mean(1)-baseline['prediction'])[mask].mean()),
                          estimated_squared_mean_drift_from_initializer=float(((x.mean(1)-baseline['prediction'])**2)[mask].mean()-correction),
                          first16_macro_rmse=macro_rmse(x[:, :16].mean(1), y, mask),
                          second16_macro_rmse=macro_rmse(x[:, 16:].mean(1), y, mask),
                          independent_half_mean_disagreement_mm=float(np.sqrt(((x[:, :16].mean(1)-x[:, 16:].mean(1))**2)[mask].mean())))
            results[family][product] = record
    rng = np.random.default_rng(20261123)
    mu, sigma, target, k, repetitions = .5, 2., 0., 4, 100000
    members = rng.normal(mu, sigma, (repetitions, k))
    error = members-target
    naive = error.mean(1)**2
    unbiased = (error.sum(1)**2-(error**2).sum(1))/(k*(k-1))
    corrected = naive-members.var(1, ddof=1)/k
    np.testing.assert_allclose(unbiased, corrected, atol=1e-12, rtol=1e-12)
    simulation = dict(mu=mu, sigma=sigma, target=target, members=k, repetitions=repetitions,
                      true_mean_mse=(mu-target)**2, theoretical_naive_mse=(mu-target)**2+sigma**2/k,
                      naive_estimate=float(naive.mean()), unbiased_estimate=float(unbiased.mean()),
                      unbiased_mean_standard_error=float(unbiased.std(ddof=1)/np.sqrt(repetitions)),
                      negative_unbiased_fraction=float((unbiased < 0).mean()),
                      identity_max_error=float(abs(unbiased-corrected).max()))
    result = dict(ensemble_audit=results, simulation=simulation,
                  scope='Archived C48 predictions only; no training, no new selection, no D labels',
                  caution='Micro-MSE correction estimates Monte Carlo error; it is not station-macro RMSE or evidence of deployment accuracy')
    (OUT/'finite_member_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
