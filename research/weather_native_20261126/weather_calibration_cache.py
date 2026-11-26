"""Frozen full-field likelihood sufficient statistics and past-only weather summaries."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from campaign import RESULTS,read,STOP
from inducing_flow_series import source_paths,LOCK
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_inducing_flow_experiment import codes,batch,require_calendar
from tools.terrain_diffusion.native_inducing_flow import InducingLatentFlow
from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import WeatherSources
from tools.terrain_diffusion.native_model_seal import validate_model_seal
from tools.terrain_diffusion.native_student_process import affine_covariance,masked_radius


def cache_codes():
    helper=Path(__file__).resolve().parents[2]/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/tools/terrain_diffusion/native_student_process.py'
    return {**codes(),'native_student_process.py':sha(helper),'cache_driver':sha(__file__)}


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],required=True);p.add_argument('--calendar-day',type=int,choices=[0,8],default=0)
    args=p.parse_args();seed=args.seed;day=args.calendar_day
    if STOP.exists():raise SystemExit(75)
    torch.set_num_threads(2);paths=source_paths(seed,day)
    root=RESULTS/f'20261126_weather_scale_tail/cache_{"calendar8" if day else "train"}_s{seed}'
    if root.exists():
        manifest=read(root/'manifest.json')
        if manifest['code_sha256']!=cache_codes() or not manifest['complete']:raise ValueError('existing cache differs or is incomplete')
        for n,v in manifest['files'].items():
            if sha(root/n)!=v:raise ValueError('cached statistic changed')
        print('Reused validated calibration cache',seed,day,flush=True);return
    source=WeatherSources(SimpleNamespace(seed=seed,device='cuda',smoke_cache=False,**paths));data=source.data
    if day:
        expected=read(LOCK)['sources'][str(seed)]['calendars']['8']
        require_calendar(data.protocol,len(data),np.asarray(data.a['query_ids']),seed,sha(paths['cache']/'manifest.json'),expected['cache_manifest_sha256'])
    elif data.development or len(data)!=3000:raise ValueError('canonical3000 training blocks required')
    model_path=RESULTS/f'20261126_inducing_latent_flow/affine_time_s{seed}'
    seal=validate_model_seal(model_path,'COMPLETE_INDUCING_FLOW',codes());protocol=read(model_path/'protocol.json')
    if protocol['source']!=source.receipt or protocol['config']['family']!='affine' or protocol['config']['domain']!='time':raise ValueError('frozen affine time source required')
    model=InducingLatentFlow('affine','time').cuda().eval();model.load_state_dict(torch.load(model_path/'last.pt',map_location='cuda',weights_only=True))
    anchors=torch.tensor(protocol['anchors'],device='cuda',dtype=torch.float64)
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'protocol.json',dict(seed=seed,calendar_day=day,source=source.receipt,source_model_seal_sha256=sha(model_path/'model_artifact_seal.json'),
        cache_manifest_sha256=sha(paths['cache']/'manifest.json'),embedding_manifest_sha256=sha(paths['embeddings']/'manifest.json'),
        code_sha256=cache_codes(),role='exposed2020 development' if day else 'training2016-2019',
        coordinates='Q137 fixed' if day else 'original fixed randomQ32 fromQ128',past_only_weather90=True,strict_oof=False,confirmation_C_D_read=False))
    N=len(data);arrays={}
    for name,shape,dtype in [('q',(N,12),'float64'),('logdet',(N,12),'float64'),('dimension',(N,12),'int16'),('weather',(N,12,90),'float32'),('origins',(N,12),'int64')]:
        arrays[name]=np.lib.format.open_memmap(root/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
    start=time.monotonic()
    for i in range(N):
        if STOP.exists():
            write_json(root/'status.json',dict(status='STOPPED_AT_BUDGET',blocks=i));raise SystemExit(75)
        b,truth,_,residual,_,_,_=batch(source,i,anchors,2,seed,not bool(day))
        C=affine_covariance(model,b,residual);e=(truth.double()-b['mean'].double()).flatten(1)
        q,ld,d=masked_radius(e,C)
        for name,value in [('q',q),('logdet',ld),('dimension',d),('weather',b['global_condition'])]:arrays[name][i]=value.cpu().numpy()
        arrays['origins'][i]=data.a['origins'][i]
        if not day:
            years=np.asarray(data.a['origins'][i]).astype('datetime64[h]').astype('datetime64[Y]').astype(int)+1970
            if not np.isin(years,[2016,2017,2018,2019]).all():raise ValueError('nontraining year entered cache')
        if i==0 or (i+1)%250==0 or i+1==N:
            progress=dict(blocks=i+1,total=N,seconds=time.monotonic()-start)
            write_json(root/'progress.json',progress);print(dict(seed=seed,day=day,**progress),flush=True)
    for a in arrays.values():a.flush()
    if not np.isfinite(arrays['weather']).all():raise ValueError('nonfinite historical weather')
    valid=np.asarray(arrays['dimension'])>0;count=np.asarray(arrays['dimension']).astype('float64')
    statistic=dict(positive_origins=int(valid.sum()),positive_blocks=int(valid.any(-1).sum()),finite_target_cells=int(count.sum()),
        constant_scale2=float(np.mean(np.asarray(arrays['q'])[valid]/count[valid])))
    if not day:
        x=np.asarray(arrays['weather'],dtype='float64').reshape(-1,90);mean=x.mean(0);std=x.std(0);std[std<1e-6]=1
        statistic.update(weather_mean=mean.tolist(),weather_std=std.tolist(),normalization_role='all training histories independent of target mask')
        expected={42:8056608,17:8056636}[seed]
        if statistic['finite_target_cells']!=expected:raise ValueError('registered label support differs')
    write_json(root/'statistics.json',statistic)
    names=[n+'.npy' for n in arrays]+['statistics.json']
    write_json(root/'manifest.json',dict(complete=True,code_sha256=cache_codes(),protocol_sha256=sha(root/'protocol.json'),files={n:sha(root/n) for n in names}))
    write_json(root/'status.json',dict(status='COMPLETE_VOLATILITY_CACHE',blocks=N,seconds=time.monotonic()-start,**statistic))
    print('Complete likelihood/weather cache',seed,day,flush=True)


if __name__=='__main__':main()
