"""Synthetic mechanism checks, not training or GNSS accuracy evidence."""
import numpy as np
import torch
from scipy.special import gamma
from pathlib import Path
from gnss_field.artifacts import write_json


def main():
    rng=np.random.default_rng(42);x=rng.normal(size=(24,32));y=2*x
    unit=lambda f:f/np.linalg.norm(f,axis=1,keepdims=True)
    gram_error=float(np.max(abs(unit(x)@unit(x).T-unit(y)@unit(y).T)))
    truth=np.linspace(20,80,24);amplitude_rmse=float(np.sqrt(np.mean((2*truth-truth)**2)))
    assert gram_error<1e-12 and amplitude_rmse>40
    n=24;distance=abs(np.arange(n)[:,None]-np.arange(n)[None])*10000.
    spatial=np.exp(-distance/100000.);z=rng.normal(size=(250000,n));z2=z*z
    scores=[];p=.5;factor=2**(p/2)*gamma((p+1)/2)/np.sqrt(np.pi)
    i,j=np.triu_indices(n,1);weights=np.exp(-distance[i,j]/50000.);weights/=weights.sum()
    for strength in (.1,.3,.6):
        covariance=(1-strength)*np.eye(n)+strength*spatial
        eigen=np.linalg.eigvalsh(covariance)
        a=np.sqrt(z2@(2*eigen)/n)
        b=np.sqrt(z2@(1+eigen)/n)
        independent=np.sqrt(z2.sum(1)*2/n)
        oracle=.5*a;wrong=b-.5*independent;delta=wrong-oracle
        true_increment_variance=2*(1-covariance[i,j]);independent_increment_variance=np.full(len(i),2.)
        mt=factor*true_increment_variance**(p/2);mf=factor*independent_increment_variance**(p/2)
        second=true_increment_variance**p*2**p*gamma((2*p+1)/2)/np.sqrt(np.pi)
        vs_oracle=np.sum(weights*(second-mt**2));vs_excess=np.sum(weights*(mf-mt)**2)
        scores.append(dict(spatial_variance_fraction=strength,expected_ES_oracle=float(oracle.mean()),
            independent_forecast_ES_excess=float(delta.mean()),ES_excess_standard_error=float(delta.std(ddof=1)/np.sqrt(len(delta))),
            ES_relative_excess_percent=float(100*delta.mean()/oracle.mean()),
            analytic_VS_oracle=float(vs_oracle),independent_forecast_VS_excess=float(vs_excess),
            VS_relative_excess_percent=float(100*vs_excess/vs_oracle)))
    # Minimal reproduction of a forward-time in-place normalization repeated
    # inside one unrolled graph, not a test of the full official EDM2 network.
    def unrolled(mutate):
        torch.manual_seed(42);w=torch.nn.Parameter(torch.randn(4,4));h=torch.randn(4,2)
        for _ in range(3):
            if mutate:
                with torch.no_grad():w.copy_(w/w.norm(dim=1,keepdim=True))
            h=(w/(w.norm(dim=1,keepdim=True)+1e-6))@h
        h.square().sum().backward()
        return bool(torch.isfinite(w.grad).all())
    try:
        unrolled(True);mutation_failure=False
    except RuntimeError as exc:
        mutation_failure='modified by an inplace operation' in str(exc)
        if not mutation_failure:raise
    assert mutation_failure and unrolled(False)
    write_json(Path(__file__).with_name('mechanism_sanity.json'),dict(
        scope='Synthetic Gaussian distributions with identical zero mean and unit marginal variance; not measured project model results',
        gram_normalized_error_after_doubling=gram_error,physical_RMSE_after_doubling_mm=amplitude_rmse,
        joint_score_comparison=scores,
        view_example=dict(prior_variance=1,observation_noise_variance=1,y1=2,y2=-1,
            one_observation_posterior_mean=1.,one_observation_posterior_variance=.5,
            two_observations_posterior_mean=1/3,two_observations_posterior_variance=1/3),
        unrolled_weight_normalization=dict(forward_inplace_reproduces_version_failure=mutation_failure,
            functional_normalization_backward_finite=True,scope='three-call minimal reproduction, not official35-call model validation'),
        interpretation='Normalized feature relation loss does not enforce millimetre amplitude; station dropout changes conditional distributions; ES and VS sensitivity must be tested separately'))
    print('SYNTHETIC_CHECKS_COMPLETE',gram_error,amplitude_rmse,scores,flush=True)


if __name__=='__main__':main()
