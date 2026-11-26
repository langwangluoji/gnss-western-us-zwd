"""Exact marginal preservation, with changed joint ensemble dependence."""
from pathlib import Path
import json
import numpy as np

def main():
    rng=np.random.default_rng(171117);cases,k,d=1024,48,4;rho=.85
    truth=np.sqrt(rho)*rng.normal(size=(cases,1))+np.sqrt(1-rho)*rng.normal(size=(cases,d))
    ensemble=np.sqrt(rho)*rng.normal(size=(cases,k,1))+np.sqrt(1-rho)*rng.normal(size=(cases,k,d))
    shuffled=ensemble.copy()
    for i in range(cases):
        for j in range(d):rng.shuffle(shuffled[i,:,j])
    def scores(x):
        first=np.mean(abs(x-truth[:,None]),axis=1)
        pair=abs(x[:,:,None,:]-x[:,None,:,:]);crps=first-pair.sum((1,2))/(2*k*(k-1))
        es=np.linalg.norm(x-truth[:,None],axis=-1).mean(1)-np.linalg.norm(x[:,:,None,:]-x[:,None,:,:],axis=-1).sum((1,2))/(2*k*(k-1))
        i,j=np.triu_indices(d,1);obs=abs(truth[:,i]-truth[:,j])**.5;forecast=(abs(x[:,:,i]-x[:,:,j])**.5).mean(1)
        return dict(crps_mean=float(crps.mean()),energy_mean=float(es.mean()),variogram_p05_mean=float(np.mean((forecast-obs)**2))),crps
    left,lc=scores(ensemble);right,rc=scores(shuffled)
    np.testing.assert_allclose(lc,rc,atol=1e-14,rtol=0)
    result=dict(synthetic_only=True,cases=cases,members=k,dimensions=d,truth_correlation=rho,matching_dependence=left,
        shuffled_dependence_same_exact_marginals=right,marginal_crps_difference=float(abs(lc-rc).max()),
        energy_crps_estimator='U-statistic off-diagonal finite-member fair estimate',variogram_estimator='Squared empirical p0.5 pair-moment difference, same K')
    Path(__file__).with_name('synthetic_dependence_scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
