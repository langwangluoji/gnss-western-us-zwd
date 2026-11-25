"""Observed standardized-error dependence by lead; exposed-calendar diagnosis only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from campaign import ROOT,RESULTS,read

sys.path.insert(0,str(ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'))
from tools.terrain_diffusion.data import sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    args=p.parse_args();seed=args.seed
    # The fresh weather-GP reference saves the unchanged S175 mean/std explicitly.
    folder=RESULTS/'20261126_weather_projective_gp'/f's{seed}'/'reference'
    lineage=read(folder/'lineage.json')
    if sha(folder/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('reference prediction changed')
    with np.load(folder/'evaluation.npz') as data:
        u=(data['truth'].astype('float64')-data['gaussian_distribution_mean'])/data['gaussian_distribution_std']
    cache=RESULTS/'20261126_native_weather_cache_v2'/f'calendar8_s{seed}'
    st=np.asarray(read(cache/'geometry.json')['dev_static']);a,b=np.triu_indices(len(st),1)
    distance=np.linalg.norm((st[a,:2]-st[b,:2])*1000,axis=-1)
    rho=read(RESULTS/'20261126_probability_context'/f'S175_s{seed}'/'copula.json')['spatial_rho']
    kernel=rho*np.exp(-.5*((distance/300)**2+((st[a,2]-st[b,2])*2)**2))
    rows=[];rng=np.random.default_rng(261126);draws=rng.integers(12,size=(2000,12))
    for lo,hi in [(0,50),(50,150),(150,300),(300,np.inf)]:
        keep=(distance>=lo)&(distance<hi);aa=a[keep];bb=b[keep];kk=kernel[keep]
        for h in range(7):
            product=np.zeros(12);square_a=np.zeros(12);square_b=np.zeros(12);counts=np.zeros(12);predicted=np.zeros(12)
            for first in range(0,len(aa),128):
                x=u[...,aa[first:first+128],h];y=u[...,bb[first:first+128],h];valid=np.isfinite(x)&np.isfinite(y)
                x=np.where(valid,x,0);y=np.where(valid,y,0);axes=(0,2,3)
                product+=(x*y).sum(axes);square_a+=x.__pow__(2).sum(axes);square_b+=y.__pow__(2).sum(axes);counts+=valid.sum(axes)
                predicted+=(valid*kk[None,None,None,first:first+128]).sum(axes)
            value=product.sum()/np.sqrt(square_a.sum()*square_b.sum())
            sampled=product[draws].sum(1)/np.sqrt(square_a[draws].sum(1)*square_b[draws].sum(1))
            rows.append(dict(distance=f'{lo}-{hi:g}',lead=h,valid_pair_hours=int(counts.sum()),
                observed_normalized_second_moment=float(value),ci95=np.quantile(sampled,[.025,.975]).tolist(),
                reference_mean_unit_variance_correlation=float(predicted.sum()/counts.sum()),
                pooled_standardized_rms=float(np.sqrt((square_a.sum()+square_b.sum())/(2*counts.sum())))))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for label,color in [('0-50','#3f8277'),('50-150','#b17b54'),('150-300','#637eac'),('300-inf','#9873a6')]:
        subset=[r for r in rows if r['distance']==label];x=np.arange(7)
        axes[0].plot(x,[r['observed_normalized_second_moment'] for r in subset],'o-',color=color,label=label+'km observed')
        axes[0].plot(x,[r['reference_mean_unit_variance_correlation'] for r in subset],'--',color=color,alpha=.8)
        axes[1].plot(x,[r['pooled_standardized_rms'] for r in subset],'o-',color=color,label=label+'km')
    axes[0].set(xlabel='Forecast lead (h)',ylabel='Normalized second moment / prior correlation',title='Solid: observed errors; dashed: fixed copula')
    axes[1].axhline(1,color='#888',ls='--');axes[1].set(xlabel='Forecast lead (h)',ylabel='Standardized error RMS',title='Marginal scale calibration shown separately')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2);ax.legend(fontsize=7)
    out=RESULTS/'20261126_weather_projective_gp'/f's{seed}'/'lead_dependence';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'lead_dependence.png',dpi=170);plt.close(fig)
    write_json(out/'lead_dependence.json',dict(rows=rows,source_prediction_sha256=lineage['prediction_sha256'],code_sha256=sha(Path(__file__)),
        interpretation='Uncentered standardized-error second moment normalized by pooled second moments; not a conditional Pearson-correlation estimate. Exposed-calendar diagnostic, shared12month bootstrap, not model inputs or a new accuracy result.',
        strict_oof=False,confirmation_C_D_read=False))
    print(json.dumps([r for r in rows if r['distance']=='0-50' and r['lead'] in (0,6)]))


if __name__=='__main__':main()
