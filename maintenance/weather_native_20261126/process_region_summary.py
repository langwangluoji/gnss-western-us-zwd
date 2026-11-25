"""Predeclared distance/coast/height/season probability strata, with paired sources."""
import argparse
import json
import numpy as np
from campaign import RESULTS,ROOT,read
from process_summary import scalar_interval
from pathlib import Path
import sys

sys.path.insert(0,str(ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'))
from tools.terrain_diffusion.data import sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42)
    args=p.parse_args();root=RESULTS/'20261126_gp_process'/f's{args.seed}';cache=RESULTS/'20261126_native_weather_cache_v2'/f'calendar8_s{args.seed}'
    static=np.asarray(read(cache/'geometry.json')['dev_static'])
    nearest=np.load(cache/'nearest.npy').reshape(3,12,72,137)
    masks={};bins=[0,50,100,150,300,np.inf]
    for lo,hi in zip(bins[:-1],bins[1:]):masks[f'distance:{lo}-{hi:g}km']=(nearest>=lo)&(nearest<hi)
    for feature,index,factor,bins in [('coast',8,500,[0,50,150,300,np.inf]),('height',2,3000,[-np.inf,500,1000,2000,np.inf])]:
        value=static[:,index]*factor
        for lo,hi in zip(bins[:-1],bins[1:]):masks[f'{feature}:{lo:g}-{hi:g}']=np.broadcast_to(((value>=lo)&(value<hi))[None,None,None],nearest.shape)
    for label,months in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]:
        masks[label]=np.broadcast_to(np.isin(np.arange(1,13),months)[None,:,None,None],nearest.shape)
    rows=[];reference=None;source=None;hashes={}
    for name in ('reference','full','local','affine'):
        folder=root/name if name=='reference' else root/name/'eval_heun0'
        lineage=read(folder/'lineage.json')
        if source is None:source=lineage['source']
        if source!=lineage['source'] or sha(folder/'evaluation.npz')!=lineage['prediction_sha256']:raise ValueError('probability lineage differs')
        hashes[name]=lineage['prediction_sha256']
        with np.load(folder/'evaluation.npz') as data:score=data['crps'].astype('float64')
        if name=='reference':reference=score.copy()
        else:np.testing.assert_array_equal(np.isfinite(score),np.isfinite(reference))
        for stratum,mask in masks.items():
            for h in (0,6):
                valid=mask&np.isfinite(score[...,h]);n=int(valid.sum())
                if not n:continue
                a=np.where(valid,score[...,h],0).sum((0,2,3));b=np.where(valid,reference[...,h],0).sum((0,2,3));count=valid.sum((0,2,3))
                active=count>0;actual=a.sum()/n;ref=b.sum()/n
                rows.append(dict(model=name,stratum=stratum,lead=h,count=n,contributing_months=int(active.sum()),
                    fair_crps_mm=float(actual),reference_fair_crps_mm=float(ref),gain_percent=float(100*(ref-actual)/ref),
                    delta_ci95_mm=scalar_interval(a[active],count[active],b[active],count[active])))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,labels,title in [(axes[0],[k for k in masks if k.startswith('distance')],'Nearest visible GNSS'),
        (axes[1],[k for k in masks if k.startswith('coast')],'Coast distance'),(axes[2],['DJF','MAM','JJA','SON'],'Season')]:
        for model,color in [('full','#b27b55'),('local','#427f79'),('affine','#617da1')]:
            values=[next((r['gain_percent'] for r in rows if r['model']==model and r['stratum']==s and r['lead']==6),np.nan) for s in labels]
            ax.plot(range(len(labels)),values,'o-',color=color,label=model)
        ax.axhline(0,color='#888',lw=.8);ax.axhline(5,color='#888',ls='--')
        ax.set(xticks=range(len(labels)),xticklabels=[x.split(':')[-1] for x in labels],ylabel='F6 fair CRPS improvement over GP (%)',title=title)
        ax.tick_params(axis='x',rotation=35);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'Seed{args.seed}: predeclared exploratory strata; paired calendar, same frozen mean and S175')
    out=root/'regions';out.mkdir(parents=True,exist_ok=True);fig.savefig(out/'probability_strata.png',dpi=170);plt.close(fig)
    write_json(out/'probability_strata.json',dict(rows=rows,source=source,source_prediction_sha256=hashes,
        interpretation='Prespecified distance/coast/height/season bins. Intervals resample common contributing months,2000 draws; no multiplicity correction. Seasonal intervals involve only3 months. Region discovery requires second seed and frozen additional-calendar verification.',
        code_sha256=sha(Path(__file__)),strict_oof=False,confirmation_C_D_read=False))
    best=sorted([r for r in rows if r['model']=='local' and r['lead']==6],key=lambda r:-r['gain_percent'])[:5]
    print(json.dumps(dict(seed=args.seed,largest_exploratory_local_F6_gains=best)))


if __name__=='__main__':main()
