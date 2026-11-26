"""Actual GNSS residual encoding audit; truncation loss is not forecast skill."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
ENGINE=ROOT/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
sys.path.insert(0,str(ENGINE))
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_weather import graph_basis,full_spectrum
from tools.terrain_diffusion.native_weather_cache import static


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(3)
    protocol=json.loads((args.cache/'protocol.json').read_text(encoding='utf-8'))
    status=json.loads((args.cache/'status.json').read_text(encoding='utf-8'));count=status['completed']
    if protocol['config']['calendar_day']:raise ValueError('this diagnostic is registered for training residuals')
    geo=dict(np.load(Path(protocol['config']['data'])/'geometry.npz'));st=static(geo)
    a={n:np.load(args.cache/f'{n}.npy',mmap_mode='r') for n in ('truth','mean','query_ids','origins')}
    # Fixed monthly midpoint: select station/time snapshots for diagnosis, never for training.
    months=np.array([str(np.datetime64(int(h),'h').astype('datetime64[M]')) for h in a['origins'][:count,0]])
    selected=[]
    for month in np.unique(months):
        indices=np.flatnonzero(months==month);selected.append(int(indices[len(indices)//2]))
    rows=[];graph_energy=[];time_energy=[]
    for i in selected:
        q=a['query_ids'][i].astype(int);s=torch.tensor(st[q]);eigen,U=graph_basis(s[:,:2]*1000,s[:,2]*3000)
        residual=torch.tensor(a['truth'][i]-a['mean'][i,:,:,None])
        if not torch.isfinite(residual).all():raise ValueError('complete native labels required for audit')
        transformed=full_spectrum(residual,U);recovered=full_spectrum(transformed,U,True)
        maximum=float((residual-recovered).abs().max());error=float((residual-recovered).square().mean().sqrt())
        total=float(transformed.square().sum());ge=transformed.square().sum((0,2)).numpy();te=transformed.square().sum((0,1)).numpy()
        graph_energy.append(ge/total);time_energy.append(te/total)
        losses={}
        for k in (16,32,64):
            partial=transformed.clone();partial[:,k:]=0;decoded=full_spectrum(partial,U,True)
            losses[str(k)]=float((decoded-residual).square().mean().sqrt())
        rows.append(dict(block=i,month=months[i],first_origin=int(a['origins'][i,0]),
            full_inverse_max_error_mm=maximum,full_inverse_rmse_mm=error,
            graph_truncation_all_horizon_rmse_mm=losses,
            first16_graph_energy_fraction=float(ge[:16].sum()/total),first2_time_energy_fraction=float(te[:2].sum()/total)))
    graph=np.mean(graph_energy,0);temporal=np.mean(time_energy,0)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    axes[0].plot(np.arange(1,len(graph)+1),np.cumsum(graph),color='#35836c');axes[0].set(xlabel='Graph modes retained',ylabel='Cumulative residual energy',ylim=(0,1),title='Full native station GFT')
    axes[1].bar(np.arange(7),temporal,color='#5986ac');axes[1].set(xlabel='7h DCT mode',ylabel='Residual energy fraction',title='Origin reconstruction + future6h')
    ks=(16,32,64);losses=[np.mean([r['graph_truncation_all_horizon_rmse_mm'][str(k)] for r in rows]) for k in ks]
    axes[2].bar([str(k) for k in ks],losses,color='#c4875d');axes[2].set(xlabel='Graph modes retained (of128)',ylabel='Encoding reconstruction RMSE (mm)',title='Truncation loses station information')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Training residual diagnostic: {len(rows)} monthly12h snapshots; not a model accuracy score')
    fig.savefig(out/'native_spectral_audit.png',dpi=170);plt.close(fig)
    payload=dict(cache_protocol_sha256=sha(args.cache/'protocol.json'),cache_status=status['status'],
        completed_blocks_available=count,monthly_snapshot_count=len(rows),rows=rows,
        average_graph_energy_fraction=graph.tolist(),average_time_energy_fraction=temporal.tolist(),
        full_inverse_max_error_mm=max(r['full_inverse_max_error_mm'] for r in rows),
        mean_truncation_rmse_mm=dict(zip(map(str,ks),losses)),
        interpretation='Exact encoding diagnostic, not hidden-station inference or diffusion training gain',
        figure_sha256=sha(out/'native_spectral_audit.png'),script_sha256=sha(Path(__file__)))
    write_json(out/'native_spectral_audit.json',payload)
    print(json.dumps({k:payload[k] for k in ('completed_blocks_available','monthly_snapshot_count','full_inverse_max_error_mm','mean_truncation_rmse_mm')}))


if __name__=='__main__':main()
