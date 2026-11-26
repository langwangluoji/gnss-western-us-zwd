"""Reproduce the finite-member interval experiment without GNSS observations."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr


def main(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261122)
    output = {}
    for k in (16, 32, 64, 128):
        sums, squares = np.zeros(3), np.zeros(3)
        for _ in range(20):
            z = rng.standard_normal((10000, k))
            for j, level in enumerate((.8, .9, .95)):
                alpha = (1-level)/2
                lo, hi = np.quantile(z, [alpha, 1-alpha], axis=1)
                probability = ndtr(hi)-ndtr(lo)
                sums[j] += probability.sum()
                squares[j] += (probability**2).sum()
        mean = sums/200000
        se = np.sqrt((squares/200000-mean**2)/200000)
        output[str(k)] = {str(level): dict(expected_coverage=float(m), simulation_se=float(s))
                          for level, m, s in zip((80, 90, 95), mean, se)}
    record = dict(draws_per_K=200000, seed=20261122,
                  distribution='N(0,1); numpy linear sample quantiles; independent target integrated analytically',
                  results=output)
    (destination/'finite_ensemble_interval_simulation.json').write_text(
        json.dumps(record, indent=2), encoding='utf-8')
    rng = np.random.default_rng(713)
    direct = {}
    for k in (16, 32):
        z, y = rng.normal(size=(100000, k)), rng.normal(size=100000)
        lo, hi = np.quantile(z, [.05, .95], axis=1)
        actual = float(np.mean((y >= lo) & (y <= hi)))
        expected = output[str(k)]['90']['expected_coverage']
        assert abs(actual-expected) < .006
        direct[str(k)] = dict(direct_coverage=actual, integrated_coverage=expected,
                              absolute_difference=abs(actual-expected))
    z = rng.random((100000, 16)); lo, hi = np.quantile(z, [.05, .95], axis=1)
    exact, approx = 15*.9/17, float(np.mean(hi-lo))
    assert abs(exact-approx) < .002
    (destination/'finite_interval_independent_checks.json').write_text(json.dumps(dict(
        independent_normal_check=direct, passed=True,
        uniform_order_statistic=dict(K=16, exact=exact, simulation=approx)), indent=2), encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)

