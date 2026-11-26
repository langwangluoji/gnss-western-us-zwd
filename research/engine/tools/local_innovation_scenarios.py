"""Fixed-cohort station-pair/scenario diagnostics and Gaussian K-prefix check."""
import json
import numpy as np
import torch
from scipy.spatial.distance import cdist
from local_innovation_pilot import Pilot,ROOT,SOURCE,local_runtime,probability_metrics
from gnss_field.artifacts import read_json,write_json
from gnss_field.data import PackedStore
from gnss_field.evaluation import point_metrics
from gnss_field.runtime import configure


def main():
    configure(type('R',(),{'cpus':2})(),42)
    data=Pilot();results=read_json(ROOT/'reviewed_results.json')
    hours=np.asarray(data.lock['outer_hours']);ix=data.indices(hours)
    store=PackedStore(local_runtime()['data_root'],purpose='context');aid=store.ids('A')
    distance=cdist(data.a['xy'],store.xy[aid])
    nearest=[];density=[]
    for h in hours:
        valid=np.isfinite(store.context_values([int(h)])[0]).all(1)
        nearest.append(np.min(distance[:,valid],axis=1))
        density.append((distance[:,valid]<=150000).sum(1))
    nearest=np.array(nearest);density=np.array(density)
    output={}
    for fold in (0,1):
        split=data.splits[fold];pred=np.load(ROOT/f'fold_{fold}/reviewed/mean.npz')['prediction'][ix]
        truth=data.a['truth'][ix];base=data.a['mean'][ix];pop=data.a['valid'][ix]&split['outer'][None]
        active=np.flatnonzero(pop.any(0));a,b=np.triu_indices(len(active),1);a,b=active[a],active[b]
        dist=np.linalg.norm(data.a['xy'][a]-data.a['xy'][b],axis=1)/1000
        selected=dist<=200;a,b,dist=a[selected],b[selected],dist[selected]
        support=pop[:,a]&pop[:,b]
        height=data.a['height'];up=np.exp(height/3000)
        y=truth[:,b]*up[b]-truth[:,a]*up[a]
        p=pred[:,b]*up[b]-pred[:,a]*up[a];old=base[:,b]*up[b]-base[:,a]*up[a]
        increments={}
        for lo,hi in [(0,25),(25,50),(50,100),(100,200)]:
            mask=support&(dist>=lo)[None]&(dist<(hi if hi<200 else 200.00001))[None]
            surface_mask=mask&(abs(height[a]-height[b])<=100)[None]
            increments[f'{lo}-{hi}']=dict(common_height=dict(base=point_metrics(y,old,mask),selected=point_metrics(y,p,mask)),
                same_height_surface=dict(base=point_metrics(truth[:,b]-truth[:,a],base[:,b]-base[:,a],surface_mask),
                    selected=point_metrics(truth[:,b]-truth[:,a],pred[:,b]-pred[:,a],surface_mask)),
                population='all outer station pairs in distance bin; pair-macro errors, not station-macro errors')
        geometric=data.a['query_weights']/np.exp(-height[:,None]/3000)
        relief=(data.a['features'][ix][:,data.a['query_indices'],53]*geometric[None]).sum(-1)*3000
        cases=dict(sparse=density<=3,dense=density>=6,input_distance_75_100km=(nearest>=75000)&(nearest<=100000),
            elevation_ge_1500m=np.broadcast_to(height[None]>=1500,pop.shape),terrain_relief_ge_500m=relief>=500)
        scenarios={name:dict(base=point_metrics(truth,base,pop&mask),selected=point_metrics(truth,pred,pop&mask)) for name,mask in cases.items()}
        mean=np.load(ROOT/f'fold_{fold}/reviewed/mean.npz')['prediction'];hi=split['inner_hours']
        k128={}
        for kind,row in results[str(fold)]['Gaussian'].items():
            score=probability_metrics(data,mean[data.indices(hi)],hi,split['inner'],row['field'],row['nugget'],members=128)
            k128[kind]=dict(group_ES_mm=score['group_ES_mm'],CRPS_mm=score['CRPS_mm'],group_VS=score['group_VS'])
        winner=min(k128,key=lambda x:k128[x]['group_ES_mm'])
        qj=(data.a['innovation'][ix][:,data.a['query_indices']]*data.a['query_weights'][None,:,:,None]).sum(2)
        reach=.5*(abs(qj[:,:,0])+abs(qj[:,:,1]))
        output[str(fold)]=dict(increments=increments,scenarios=scenarios,Gaussian_inner_K128=k128,
            Gaussian_K64_selected=results[str(fold)]['probability_selected'],Gaussian_K128_diagnostic_best=winner,
            Gaussian_prefix_rank_stable=winner==results[str(fold)]['probability_selected'],
            innovation_query_RMS_mm=float(np.sqrt(np.mean(qj[pop]**2))),
            innovation_envelope_proxy_percentiles_mm=np.quantile(reach[pop],[.1,.5,.9]).tolist(),
            selected_correction_RMS_mm=float(np.sqrt(np.mean((pred-base)[pop]**2))))
    write_json(ROOT/'scenario_diagnostics.json',output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='DejaVu Sans'
    field=np.load(ROOT/'fold_0/reviewed/publication_case.npz');land=field['land'].astype(bool)
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,key,title in zip(axes,['base_Z0','mean_Z0','correction_Z0'],['Frozen M0+IDW','Selected matched mean','Learned correction']):
        value=np.where(land,field[key],np.nan)
        kwargs={}
        if key=='correction_Z0':
            limit=max(float(np.nanmax(abs(value))),1e-8);kwargs=dict(cmap='RdBu_r',vmin=-limit,vmax=limit)
        im=ax.imshow(value,origin='lower',**kwargs);ax.set(title=title);ax.axis('off');fig.colorbar(im,ax=ax,shrink=.7,label='Z0 mm')
    fig.suptitle('First registered outer hour: 25 km output spacing; field truth unavailable')
    fig.savefig(ROOT/'publication_field.png',dpi=150);plt.close(fig)
    print(json.dumps({f:dict(correction_RMS_mm=r['selected_correction_RMS_mm'],K_stable=r['Gaussian_prefix_rank_stable']) for f,r in output.items()}),flush=True)


if __name__=='__main__':main()
