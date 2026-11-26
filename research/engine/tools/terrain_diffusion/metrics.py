"""Metrics in millimetres; arrays are [member,time,station]."""
import numpy as np


def evaluate(samples,truth):
    samples=np.asarray(samples);truth=np.asarray(truth)
    valid=np.isfinite(truth)
    if not valid.any():raise ValueError('No valid truth in scoring population')
    if not np.isfinite(samples[:,valid]).all():raise ValueError('Nonfinite prediction in fixed truth population')
    error=np.where(valid,samples.mean(0)-truth,np.nan)
    rmse=np.sqrt(np.nanmean(error**2,axis=0))
    k=len(samples)
    first=np.mean(np.abs(samples-truth[None]),axis=0)
    pair=np.abs(samples[:,None]-samples[None,:]).sum((0,1))
    crps=first-pair/(2*k*(k-1)) if k>1 else first
    lo,hi=np.quantile(samples,[.05,.95],axis=0)
    return dict(station_macro_rmse_mm=float(np.nanmean(rmse)),micro_rmse_mm=float(np.sqrt(np.nanmean(error**2))),
                mae_mm=float(np.nanmean(abs(error))),bias_mm=float(np.nanmean(error)),
                fair_crps_mm=float(np.nanmean(np.nanmean(np.where(valid,crps,np.nan),axis=0))),coverage90=float(np.mean(((truth>=lo)&(truth<=hi))[valid])),
                p90_absolute_mm=float(np.nanquantile(abs(error),.9)),p95_absolute_mm=float(np.nanquantile(abs(error),.95)),negative_fraction=float(np.mean(samples<0)),
                width90_mm=float(np.mean((hi-lo)[valid])),valid_count=int(valid.sum()),station_count=int(np.isfinite(rmse).sum()))
