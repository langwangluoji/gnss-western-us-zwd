"""Actual canonical GNSS4-update CPU spike; no development labels or ranking."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from campaign import RESULTS,read,STOP
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_paths import file_path
from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import WeatherSources
from tools.terrain_diffusion.native_weather_gp_experiment import codes as gp_codes
from tools.terrain_diffusion.native_model_seal import validate_model_seal
from tools.terrain_diffusion.native_weather_gp_diffusion import condition265
from tools.terrain_diffusion.native_weather_gp import weather_summary
from tools.terrain_diffusion.native_latent_flow import ProjectiveLatentFlow,joint_score


def pairs_for_geometry(static,rng):
    a,b=np.triu_indices(len(static),1);xy=static[:,:2].numpy()*1000
    distance=np.linalg.norm(xy[a]-xy[b],axis=-1);selected=[]
    for lo,hi in [(0,50),(50,150),(150,300),(300,np.inf)]:
        pool=np.flatnonzero((distance>=lo)&(distance<hi));selected.extend(rng.permutation(pool)[:4].tolist())
    return torch.tensor(np.stack([a[selected],b[selected]],-1),dtype=torch.long)


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42);args=p.parse_args();seed=args.seed
    if STOP.exists():raise SystemExit(75)
    torch.set_num_threads(2)
    gp=RESULTS/f'20261126_weather_projective_gp/s{seed}/c265';gp_protocol=read(gp/'protocol.json')
    cfg=gp_protocol['config'];source_args=SimpleNamespace(seed=seed,device='cpu',smoke_cache=True,gp=gp,
        **{k:file_path(cfg[k]) for k in ['cache','embeddings','gaussian','copula']})
    source=WeatherSources(source_args)
    validate_model_seal(gp,'COMPLETE_WEATHER_GP',gp_codes())
    if source.data.development:raise ValueError('engineering updates require training data')
    root=RESULTS/f'20261126_projective_latent_flow_spike/s{seed}';root.mkdir(parents=True,exist_ok=False)
    source_receipt={**source.receipt,'cache_manifest_sha256':sha(source_args.cache/'manifest.json'),
        'embedding_manifest_sha256':sha(source_args.embeddings/'manifest.json')}
    code=Path(__file__).resolve().parents[2]/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/tools/terrain_diffusion/native_latent_flow.py'
    write_json(root/'protocol.json',dict(seed=seed,steps=4,members=8,queries=32,device='cpu',source=source_receipt,
        source_code_sha256=sha(code),driver_sha256=sha(__file__),mean='frozen TF3000; no S175 center shift',
        std='frozen S175 training scale times conditional gs',objective='joint proper scores; endpoint probability flow, not denoising matching',
        experimental_mode='ENGINEERING_ONLY',strict_oof=False,confirmation_C_D_read=False))
    traces={};initial_losses=[];parameters={};start=time.monotonic();covariance_difference=None
    for family in ['affine','odd']:
        for domain in ['time','frequency']:
            torch.manual_seed(seed);model=ProjectiveLatentFlow(family,domain)
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            torch.manual_seed(seed+261126);trace=[];label=family+'_'+domain
            parameters[label]=sum(p.numel() for p in model.parameters())
            for step in range(4):
                if STOP.exists():raise SystemExit(75)
                with torch.no_grad():
                    c,gm,gs,mean,ids,static=source.batch(step);inputs,truth,_,_=source.data.batch(step,72)
                    history=inputs['history'];rng=np.random.default_rng(seed+261126+step)
                    index=torch.tensor(np.sort(rng.choice(c.shape[1],32,replace=False)),dtype=torch.long)
                    correlation=source.weather_gp.correlation(c[:,index],history,static[index],ids[index])
                    if covariance_difference is None:
                        full=source.weather_gp.correlation(c[:1],history[:1],static,ids)
                        flat=(index[:,None]*7+torch.arange(7)[None]).flatten()
                        covariance_difference=float((full[:,flat][:,:,flat]-correlation[:1]).abs().max())
                        if covariance_difference>2e-6:raise ValueError('GP query-subset law changed')
                    root_R=torch.linalg.cholesky(correlation)
                    K=8;B=len(c);eta=torch.randn(K,B,len(index)*7,dtype=torch.float64)
                    gp_path=torch.einsum('bij,kbj->kbi',root_R,eta).reshape(K,B,len(index),7).float()
                    latent_noise=torch.randn(K,B,8,7)
                    condition=condition265(c[:,index],history);global_condition=weather_summary(history)
                    mean=mean[:,index];std=source.scale*gs[:,index];truth=truth[:,index]
                    pairs=pairs_for_geometry(static[index],rng)
                samples=model(condition,global_condition,mean,std,gp_path,latent_noise)
                if step==0:
                    torch.testing.assert_close(samples,mean[None]+std[None]*gp_path,rtol=0,atol=0)
                loss,terms=joint_score(samples,truth,mean,std,correlation,pairs)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite projective-flow loss')
                optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                if not torch.isfinite(norm):raise FloatingPointError('nonfinite projective-flow gradient')
                optimizer.step()
                row=dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm),
                    terms={k:float(v.detach()) for k,v in terms.items()},finite_labels=int(torch.isfinite(truth).sum()),
                    first_origin_hour=int(source.data.a['origins'][step,0]),pair_count=len(pairs),
                    phi_readout_norm=float(model.query.net[-1].weight[1:].detach().norm()),seconds=time.monotonic()-start)
                trace.append(row);print(label,row,flush=True)
            initial_losses.append(trace[0]['loss']);traces[label]=trace
            torch.save(model.state_dict(),root/f'{label}_step4.pt')
    if max(initial_losses)-min(initial_losses)>1e-10:raise ValueError('initial common-noise scores differ')
    write_json(root/'summary.json',dict(complete=True,traces=traces,parameters=parameters,
        initial_common_noise_loss_exact=True,gp_subset_max_difference=covariance_difference,total_seconds=time.monotonic()-start,
        role='engineering only; actual continuous four training blocks; no development scoring',
        protocol_sha256=sha(root/'protocol.json'),strict_oof=False,confirmation_C_D_read=False))
    print('Complete actual4-update projective latent-flow spike',flush=True)


if __name__=='__main__':main()
