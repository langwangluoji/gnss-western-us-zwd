"""Full2024 fixed-key point and IID coupled-field probability evaluation."""
import argparse
import json

import numpy as np
import torch

from .features import HERE,CACHE,PARENT,OLD_CACHE,sha,write_json
from .prob_train import Data,destination,load_model,codes
from tools.terrain_diffusion.native_weather_evaluation import fair_crps,gaussian_crps
from tools.terrain_diffusion.native_copula import spatial_correlation
from tools.terrain_diffusion.native_weather_gp_diffusion import trajectory_factors,transform_gp_path,affine_marginals

KINDS={'base':('gaussian','gp','local'),'vmf':('gaussian','gp','affine','local')}


def elementary_scores(samples,mean,truth,valid):
    y=np.where(valid,truth,np.nan);error=np.where(valid,mean-truth,np.nan)
    crps=fair_crps(samples,y);sample_error=np.where(valid,samples.mean(0)-truth,np.nan)
    lo,hi=np.quantile(samples,[.1,.9],axis=0);count=valid.sum((0,1))
    result=dict(count_by_lead=count.tolist(),point_rmse_mm_by_lead=np.sqrt(np.nanmean(error**2,(0,1))).tolist(),sample_mean_rmse_mm_by_lead=np.sqrt(np.nanmean(sample_error**2,(0,1))).tolist(),fair_crps_mm_by_lead=np.nanmean(crps,(0,1)).tolist(),empirical_q10_q90_coverage=(((truth>=lo)&(truth<=hi)&valid).sum((0,1))/count).tolist())
    if len(samples)==32:
        lower,upper=np.sort(samples,axis=0)[[2,29]]
        result['rank3_30_coverage']=(((truth>=lower)&(truth<=upper)&valid).sum((0,1))/count).tolist();result['rank3_30_exchangeable_reference']=27/33
    if truth.shape[-1]>1:
        inc_valid=valid[...,1:]&valid[...,:-1];target=np.where(inc_valid,np.diff(truth,axis=-1),np.nan)
        result['six_hour_increment_fair_crps_mm']=float(np.nanmean(fair_crps(np.diff(samples,axis=-1),target)))
    return result


def seal():
    path=HERE/'probability_lock.json'
    if path.exists():raise FileExistsError(path)
    weights={}
    for seed in (42,17):
        for view,kinds in KINDS.items():
            for kind in kinds:
                folder=destination(seed,view,kind);status=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
                if status['steps']!=3000 or status['supervised_updates']!=3000 or status['optimizer_state_steps']!=[3000]:raise ValueError('incomplete formal probability model')
                load_model(seed,view,kind)
                weights[f'{seed}/{view}/{kind}']=dict(weight_sha256=sha(folder/'last.pt'),protocol_sha256=sha(folder/'lock.json'))
    write_json(path,dict(code_sha256=codes(),models=weights,members=32,solver='heun0',sampling_steps=20,sample_seed=261126,primary_cohort='previous confirm common63494 anchors; all future horizons score where observed',selection='registered final3000; no2024 model selection',source_seals={str(s):sha(HERE/f'diffusion_source_s{s}.json') for s in (42,17)},point_mean_selection_sha256=sha(HERE/'selection_2023.json')))


@torch.no_grad()
def evaluate(seed,view,kind):
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.45)
    lock=json.loads((HERE/'probability_lock.json').read_text(encoding='utf-8'))
    if lock['code_sha256']!=codes():raise ValueError('probability code changed after lock')
    data=Data(seed);cache=dict(np.load(data.root/'test.npz'));first=int(cache['fit_start']);pos=np.arange(first,len(cache['mean']));q=cache['query'];T=len(pos);Q=len(q);K=32
    g=load_model(seed,view,'gaussian');gp=None if kind=='gaussian' else load_model(seed,view,'gp');flow=load_model(seed,view,kind) if kind in ('affine','local') else None
    copula=json.loads((destination(seed,view,'gaussian')/'copula.json').read_text(encoding='utf-8'))
    samples=np.empty((K,T,Q,7),dtype='f4');analytic_mean=np.empty((T,Q,7),dtype='f4');analytic_std=np.empty_like(analytic_mean)
    latent_dir=CACHE/f'coupled_noise_s{seed}_{view}';latent=None;local_roots=None
    if kind=='gp':
        latent_dir.mkdir(exist_ok=False)
        latent=np.lib.format.open_memmap(latent_dir/'path.npy',mode='w+',dtype='float64',shape=(K,T,Q,7))
        local_roots=np.lib.format.open_memmap(latent_dir/'roots.npy',mode='w+',dtype='float64',shape=(T,Q,7,7))
    elif flow is not None:
        receipt=json.loads((latent_dir/'receipt.json').read_text(encoding='utf-8'))
        expected=lock['models'][f'{seed}/{view}/gp']['weight_sha256']
        if receipt['gp_sha256']!=expected or receipt['source_sha256']!=sha(HERE/f'diffusion_source_s{seed}.json'):raise ValueError('coupled-noise source differs')
        for name,value in receipt['files'].items():
            if sha(latent_dir/name)!=value:raise ValueError('coupled noise cache changed')
        latent=np.load(latent_dir/'path.npy',mmap_mode='r');local_roots=np.load(latent_dir/'roots.npy',mmap_mode='r')
    static=torch.tensor(data.static[q],device='cuda');space=torch.linalg.cholesky(spatial_correlation(static,copula['spatial_rho']).double());time=torch.linalg.cholesky(torch.tensor(copula['time_correlation'],device='cuda',dtype=torch.float64))
    torch.manual_seed(lock['sample_seed']);out=HERE/'evaluation'/f's{seed}'/view/kind;out.mkdir(parents=True,exist_ok=False)
    for begin in range(0,T,4):
        p=pos[begin:begin+4];c,summary,mean,truth,st,ids=data.arrays(cache,p,view);B=len(p);gm,logs=g(c);center=mean+data.bias+data.scale*gm;std=data.scale*logs.exp()
        noise=torch.randn(K,B,Q,7,device='cuda').double() if flow is None else None
        if flow is not None:
            path=torch.tensor(np.asarray(latent[:,begin:begin+B]),device='cuda')
        elif gp is None:
            path=torch.einsum('qr,kbrh->kbqh',space,noise)@time.T
        else:
            R=gp.correlation(c,summary,st,ids);root=torch.linalg.cholesky(R)
            path=torch.einsum('bij,kbj->kbi',root,noise.reshape(K,B,Q*7)).reshape(K,B,Q,7)
            latent[:,begin:begin+B]=path.cpu().numpy()
            root_time,_=trajectory_factors(gp.query_time_blocks(c,summary,st,ids));local_roots[begin:begin+B]=root_time.cpu().numpy()
        am=center;ast=std
        if flow is not None:
            root=torch.tensor(np.asarray(local_roots[begin:begin+B]),device='cuda')
            full=torch.cat([c,summary[:,None].expand(-1,Q,-1)],-1)
            path=transform_gp_path(flow,full.repeat(K,1,1),path.reshape(K*B,Q,7),root.repeat(K,1,1,1),20,'heun0').reshape(K,B,Q,7)
            if kind=='affine':am,ast=affine_marginals(flow,full,root,center,std,20,'heun0')
        physical=center[None].double()+std[None].double()*path
        if not torch.isfinite(physical).all():raise FloatingPointError('nonfinite coupled diffusion sample')
        samples[:,begin:begin+B]=physical.float().cpu().numpy();analytic_mean[begin:begin+B]=am.cpu().numpy();analytic_std[begin:begin+B]=ast.cpu().numpy()
        if begin==0 or (begin//4+1)%50==0:print(json.dumps(dict(seed=seed,view=view,kind=kind,completed=begin+B,total=T)),flush=True)
    if kind=='gp':
        latent.flush();local_roots.flush()
        write_json(latent_dir/'receipt.json',dict(gp_sha256=lock['models'][f'{seed}/{view}/gp']['weight_sha256'],source_sha256=sha(HERE/f'diffusion_source_s{seed}.json'),sample_seed=261126,members=32,files={name:sha(latent_dir/name) for name in ('path.npy','roots.npy')},purpose='identical frozen GP paths for matched Gaussian/affine/local comparison'))
    truth=cache['truth'][first:];mean=cache['mean'][first:];parent=np.load(OLD_CACHE/'comparison_arrays.npz');common=parent['valid'][:,q]
    if not np.array_equal(data.hours[pos+int(cache['start'])],parent['hours']):raise ValueError('evaluation calendar differs from fixed anchors')
    roles=np.load(OLD_CACHE/'data.npz')['role'][q];valid=np.isfinite(truth)&common[:,:,None];primary=valid&(roles=='confirm')[None,:,None]
    metrics=elementary_scores(samples,mean,truth,primary)
    if kind!='local':metrics['analytic_gaussian_crps_mm_by_lead']=np.nanmean(np.where(primary,gaussian_crps(analytic_mean,analytic_std,truth),np.nan),(0,1)).tolist()
    if kind!='local':metrics['analytic_gaussian80_coverage']=(((abs((truth-analytic_mean)/analytic_std)<=1.2815515655446004)&primary).sum((0,1))/primary.sum((0,1))).tolist()
    # Paired unbiased energy estimate on the same observed field cells.
    clean=np.where(primary[None],samples-truth[None],0).astype('f8');dimension=primary.sum((1,2));den=np.sqrt(np.maximum(dimension,1))
    first_term=np.sqrt((clean*clean).sum((2,3))).mean(0)/den
    difference=np.where(primary[None],samples[:16]-samples[16:],0).astype('f8');second=np.sqrt((difference*difference).sum((2,3))).mean(0)/den
    energy=np.where(dimension>0,first_term-.5*second,np.nan);metrics['paired_energy_mm']=float(np.nanmean(energy));metrics['negative_sample_fraction']=float((samples<0).mean())
    score=fair_crps(samples,np.where(valid,truth,np.nan)).astype('f4')
    np.savez_compressed(out/'evaluation.npz',samples=samples,truth=truth,mean=mean,valid=valid,primary=primary,crps=score,query=q,hours=parent['hours'],energy=energy,analytic_mean=analytic_mean,analytic_std=analytic_std)
    write_json(out/'metrics.json',metrics);write_json(out/'receipt.json',dict(probability_lock_sha256=sha(HERE/'probability_lock.json'),source_sha256=sha(HERE/f'diffusion_source_s{seed}.json'),sample_seed=261126,members=32,solver='heun0',steps=20,independent_iid_member_noise=True,weights=lock['models'][f'{seed}/{view}/{kind}'],prediction_sha256=sha(out/'evaluation.npz'),scores_sha256=sha(out/'metrics.json'),deterministic_mean_preserved=True))
    print(json.dumps(dict(seed=seed,view=view,kind=kind,metrics=metrics)),flush=True)


def run(seed):
    for view,kinds in KINDS.items():
        for kind in kinds:evaluate(seed,view,kind)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    if args.seal:seal()
    else:run(args.seed)
