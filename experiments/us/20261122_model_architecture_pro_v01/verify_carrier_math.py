"""Gauss-Hermite quadrature checks for a proposed distributional head, not training."""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss


def deform(x, y, b):
    return (x+(y+b*(y*y-1)/np.sqrt(2))/np.sqrt(1+b*b))/np.sqrt(2)


def covariance(rho, b1, b2):
    return .5*(rho+(rho+b1*b2*rho*rho)/np.sqrt((1+b1*b1)*(1+b2*b2)))


def main():
    nodes, weights = hermgauss(16)
    nodes, weights = np.sqrt(2)*nodes, weights/np.sqrt(np.pi)
    x, y = nodes[:, None], nodes[None, :]
    ww = weights[:, None]*weights[None, :]
    records = {}
    for b in (0., .5, -.5, 1., -1., 3.):
        r = deform(x, y, b)
        mean = np.sum(ww*r); variance = np.sum(ww*r*r)
        skew = np.sum(ww*r**3)
        excess = np.sum(ww*r**4)-3
        expected_skew = b*(3+2*b*b)/(2*(1+b*b)**1.5)
        expected_excess = 3*b*b*(2+b*b)/(1+b*b)**2
        assert abs(mean) < 1e-12 and abs(variance-1) < 1e-12
        assert abs(skew-expected_skew) < 1e-11
        assert abs(excess-expected_excess) < 1e-11
        records[str(b)] = dict(mean=float(mean), variance=float(variance),
                              skewness=float(skew), excess_kurtosis=float(excess))
    u1, u2, v1, v2 = np.meshgrid(nodes, nodes, nodes, nodes, indexing='ij')
    w4 = weights[:, None, None, None]*weights[None, :, None, None]*weights[None, None, :, None]*weights[None, None, None, :]
    cov_checks = []
    for rho in (.65, -.4):
        b1, b2 = .5, -.8
        r1 = deform(u1, v1, b1)
        r2 = deform(rho*u1+np.sqrt(1-rho*rho)*u2,
                    rho*v1+np.sqrt(1-rho*rho)*v2, b2)
        actual, expected = float(np.sum(w4*r1*r2)), covariance(rho, b1, b2)
        assert abs(actual-expected) < 1e-11
        cov_checks.append(dict(rho=rho, quadrature=actual, formula=expected))
    rng = np.random.default_rng(42)
    p, q = rng.normal(size=(2, 10000))
    exact_init = deform((p+q)/np.sqrt(2), (p-q)/np.sqrt(2), 0.)
    initialization_error = float(np.max(abs(exact_init-p)))
    assert initialization_error < 1e-12
    rho, b = .3, .8
    grid_query_variance = .5*(1+covariance(rho, b, b))
    transform_query_variance = .5*(1+rho)
    assert abs(grid_query_variance-transform_query_variance) > .01
    dependent_a_mean = float(np.sum(weights*nodes*np.exp(.3*nodes)))
    assert abs(dependent_a_mean-.3*np.exp(.5*.3**2)) < 1e-12
    assert dependent_a_mean > .3  # Positive scale dependent on carrier destroys zero mean.
    result = dict(passed=True, integration='16-point Gauss-Hermite per Gaussian dimension',
        marginal_moments=records, spatial_covariance=cov_checks,
        step0_rotation_max_error=initialization_error,
        nonlinear_query_counterexample=dict(grid_then_query_variance=grid_query_variance,
                                            query_then_nonlinearity_variance=transform_query_variance),
        carrier_dependent_coefficient_counterexample_mean=dependent_a_mean,
        scope='Mathematics only. No learned accuracy, physical truth or latent identifiability claim.')
    Path(__file__).with_name('carrier_math_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
