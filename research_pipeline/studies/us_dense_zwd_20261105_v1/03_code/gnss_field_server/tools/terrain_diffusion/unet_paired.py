"""Paired development-calendar day-block bootstrap; seeds are averaged, not resampled."""
import json
import numpy as np
from .unet_foundation import ROOT
from .residual_experiment import load_pack
from .data import write_json

def run():
    rng=np.random.default_rng(1313);draws=rng.integers(0,12,(1000,12));weights=np.stack([np.bincount(x,minlength=12) for x in draws]);values={};means={}
    for arm in sorted(ROOT.glob('E130[1-4]*')):
        boot=[];original=[]
        for seed in (42,17):
            folder=arm/f's{seed}';p=load_pack(folder/'expanded/evaluation.npz');err=p['samples'][0].astype('f8')-p['truth'];ss=[];counts=[]
            for li in range(3):
                ss.append([np.nansum(err[(p['layout']==li)&(p['row']==m)]**2,axis=0) for m in range(12)]);counts.append([np.isfinite(err[(p['layout']==li)&(p['row']==m)]).sum(0) for m in range(12)])
            ss=np.asarray(ss);counts=np.asarray(counts);r=np.nanmean(np.sqrt(ss.sum(1)/counts.sum(1)),axis=1).mean();m=json.loads((folder/'expanded/metrics.json').read_text())['macro_rmse_mm'];assert abs(r-m)<2e-5;original.append(r)
            with np.errstate(invalid='ignore',divide='ignore'):
                b=np.sqrt(np.einsum('bm,lms->bls',weights,ss)/np.einsum('bm,lms->bls',weights,counts));boot.append(np.nanmean(b,axis=2).mean(1))
        values[arm.name]=np.mean(boot,axis=0);means[arm.name]=float(np.mean(original))
    base=values['E1301_mlp'];out=[]
    for arm in values:
        if arm=='E1301_mlp':continue
        diff=values[arm]-base;out.append(dict(experiment=arm,macro_rmse_mm=means[arm],difference_from_mlp_mm=means[arm]-means['E1301_mlp'],paired_development_interval95_mm=np.quantile(diff,[.025,.975]).tolist()))
    write_json(ROOT/'paired_development_bootstrap.json',dict(comparisons=out,resamples=1000,seed=1313,unit='12 sampled month-start days, shared across layouts and models; two model seeds averaged within each replicate',interpretation='Exploratory paired development-calendar uncertainty, not blind/full-year confirmation; does not treat layouts, station-hours or two optimizer seeds as independent weather samples'))
    print(json.dumps(out,indent=2))

if __name__=='__main__':run()
