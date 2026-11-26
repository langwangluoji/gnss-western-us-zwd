"""Declared-cohort finite-ensemble diagnostics, never filtering invalid predictions."""
import numpy as np
import torch
from scipy.spatial.distance import cdist
from gnss_field.evaluation import point_metrics,crps_ensemble,variogram_score
from gnss_field.objectives.scores import energy_score


def macro(values,mask):
    counts=mask.sum(0); use=counts>0
    if not use.any(): raise ValueError('No score population')
    return float((np.where(mask,values,0).sum(0)[use]/counts[use]).mean())


def score_ensemble(truth,samples,mean,population,xy):
    truth=np.asarray(truth,dtype='f8'); samples=np.asarray(samples,dtype='f8')
    population=np.asarray(population,bool); mean=np.asarray(mean,dtype='f8')
    if not np.isfinite(samples).all() or not np.isfinite(mean).all() or not np.isfinite(truth[population]).all():
        raise ValueError('Nonfinite forecast or declared observation; cannot drop score rows')
    if (population.sum(1)==0).any(): raise ValueError('Empty scored hour')
    result=point_metrics(truth,mean,population)
    clean=np.where(population,truth,0.)
    result['CRPS_mm']=macro(crps_ensemble(clean,samples,axis=1),population)
    for level in (80,90,95):
        alpha=(100-level)/200
        lo,hi=np.quantile(samples,[alpha,1-alpha],axis=1)
        result[f'coverage{level}']=macro((truth>=lo)&(truth<=hi),population)
        result[f'width{level}_mm']=macro(hi-lo,population)
    result['ES_mm']=float(energy_score(torch.tensor(samples),torch.tensor(clean),torch.tensor(population)))
    _,groups=np.unique(np.floor(np.asarray(xy)/200000).astype(int),axis=0,return_inverse=True)
    scores=[]; variograms=[]; shuffled=[]; rng=np.random.default_rng(20261121)
    for group in np.unique(groups):
        selected=np.flatnonzero(groups==group)
        if len(selected)<4: continue
        es=[]; vs=[]; ss=[]
        for t in range(len(truth)):
            ids=selected[population[t,selected]]
            if len(ids)<4: continue
            y=truth[t,ids]; z=samples[t][:,ids]
            es.append(float(energy_score(torch.tensor(z[None]),torch.tensor(y[None]))))
            vs.append(variogram_score(y,z,cdist(xy[ids],xy[ids])/1000))
            zp=np.stack([rng.permutation(z[:,i]) for i in range(len(ids))],1)
            ss.append(float(energy_score(torch.tensor(zp[None]),torch.tensor(y[None]))))
        if es: scores.append(np.mean(es)); variograms.append(np.mean(vs)); shuffled.append(np.mean(ss))
    if not scores: raise ValueError('No preregistered group has four valid observations')
    result.update(group_ES_mm=float(np.mean(scores)),group_VS=float(np.mean(variograms)),
        shuffled_group_ES_mm=float(np.mean(shuffled)),eligible_groups=len(scores),members=samples.shape[1],
        ES_estimator='off-diagonal U statistic / sqrt(valid N)',CRPS_estimator='finite ensemble V statistic')
    return result
