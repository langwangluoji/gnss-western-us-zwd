"""Real GNSS coupled-innovation four-update CPU validation, without development scoring."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from campaign import RESULTS,read,STOP
from latent_flow_spike import pairs_for_geometry
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_paths import file_path
from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import WeatherSources
from tools.terrain_diffusion.native_weather_gp_experiment import codes as gp_codes
from tools.terrain_diffusion.native_model_seal import validate_model_seal
from tools.terrain_diffusion.native_weather_gp_diffusion import condition265
from tools.terrain_diffusion.native_weather_gp import weather_summary
from tools.terrain_diffusion.native_inducing_flow import InducingLatentFlow,anchor_locations,inducing_decomposition
from tools.terrain_diffusion.native_latent_flow import joint_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,17],default=42);seed=p.parse_args().seed
    if STOP.exists():raise SystemExit(75)
    torch.set_num_threads(2)
    gp=RESULTS/f'20261126_weather_projective_gp/s{seed}/c265';cfg=read(gp/'protocol.json')['config']
    source_args=SimpleNamespace(seed=seed,device='cpu',smoke_cache=True,gp=gp,
        **{k:file_path(cfg[k]) for k in ['cache','embeddings','gaussian','copula']})
    source=WeatherSources(source_args);validate_model_seal(gp,'COMPLETE_WEATHER_GP',gp_codes())
    if source.data.development:raise ValueError('only training data allowed')
    root=RESULTS/f'20261126_inducing_latent_flow_spike/s{seed}';root.mkdir(parents=True,exist_ok=False)
    anchors=anchor_locations(source.data.train_static,8)
    engine=Path(__file__).resolve().parents[2]/'research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server'
    names=['native_latent_flow.py','native_inducing_flow.py','native_weather_gp.py']
    receipt={**source.receipt,'cache_manifest_sha256':sha(source_args.cache/'manifest.json'),
        'embedding_manifest_sha256':sha(source_args.embeddings/'manifest.json')}
    write_json(root/'protocol.json',dict(seed=seed,steps=4,members=8,queries=32,device='cpu',source=receipt,
        code_sha256={n:sha(engine/'tools/terrain_diffusion'/n) for n in names},driver_sha256=sha(__file__),
        pair_driver_sha256=sha(Path(__file__).with_name('latent_flow_spike.py')),anchors=anchors.tolist(),
        mean='frozen TF3000; no S175 mean shift',std='frozen S175 scale times conditional gs',
        noise='independent eta and residual GP; gp_path=residual+W eta; latent eta is coupled',
        objective='joint proper-score endpoint probability flow; not denoising matching',
        experimental_mode='ENGINEERING_ONLY',strict_oof=False,confirmation_C_D_read=False))
    traces={};initial_losses=[];parameters={};start=time.monotonic();subset_error=None
    for family in ['affine','odd']:
        for domain in ['time','frequency']:
            torch.manual_seed(seed);model=InducingLatentFlow(family,domain)
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            torch.manual_seed(seed+261126);label=family+'_'+domain;trace=[]
            parameters[label]=sum(p.numel() for p in model.parameters())
            for step in range(4):
                if STOP.exists():raise SystemExit(75)
                with torch.no_grad():
                    c,gm,gs,mean,ids,static=source.batch(step);inputs,truth,_,_=source.data.batch(step,72)
                    history=inputs['history'];rng=np.random.default_rng(seed+261126+step)
                    index=torch.tensor(np.sort(rng.choice(c.shape[1],32,replace=False)),dtype=torch.long)
                    R,W,residual=inducing_decomposition(source.weather_gp,c[:,index],history,static[index],ids[index],anchors)
                    if subset_error is None:
                        full,fullW,_=inducing_decomposition(source.weather_gp,c[:1],history[:1],static,ids,anchors)
                        flat=(index[:,None]*7+torch.arange(7)[None]).flatten()
                        subset_error=max(float((full[:,flat][:,:,flat]-R[:1]).abs().max()),float((fullW[:,flat]-W[:1]).abs().max()))
                        if subset_error>2e-6:raise ValueError('query-subset law changed')
                    K=8;B=len(c);eta=torch.randn(K,B,8,7)
                    residual_noise=torch.randn(K,B,len(index)*7,dtype=torch.float64)
                    gp_flat=torch.einsum('bij,kbj->kbi',torch.linalg.cholesky(residual),residual_noise)
                    gp_flat+=torch.einsum('bnd,kbd->kbn',W,eta.double().flatten(2))
                    gp_path=gp_flat.reshape(K,B,len(index),7).float()
                    condition=condition265(c[:,index],history);global_condition=weather_summary(history)
                    mean=mean[:,index];std=source.scale*gs[:,index];truth=truth[:,index]
                    pairs=pairs_for_geometry(static[index],rng)
                samples=model(condition,global_condition,mean,std,gp_path,eta,W.float())
                if step==0:torch.testing.assert_close(samples,mean[None]+std[None]*gp_path,rtol=0,atol=0)
                loss,terms=joint_score(samples,truth,mean,std,R,pairs)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite inducing-flow loss')
                optimizer.zero_grad();loss.backward()
                flow_norm=sum(float(p.grad.square().sum()) for p in model.flow.parameters() if p.grad is not None)**.5
                phi_norm=float(model.query.net[-1].weight.grad[1:].norm())
                if step==0 and flow_norm<=0:raise ValueError('initial latent-flow gradient is blocked')
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                if not torch.isfinite(norm):raise FloatingPointError('nonfinite inducing-flow gradient')
                optimizer.step()
                row=dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm),flow_gradient_norm=flow_norm,
                    phi_gradient_norm=phi_norm,terms={k:float(v.detach()) for k,v in terms.items()},
                    finite_labels=int(torch.isfinite(truth).sum()),first_origin_hour=int(source.data.a['origins'][step,0]),
                    pair_count=len(pairs),inducing_variance_fraction=float(W.square().sum(-1).mean()),seconds=time.monotonic()-start)
                trace.append(row);print(label,row,flush=True)
            initial_losses.append(trace[0]['loss']);traces[label]=trace
            torch.save(model.state_dict(),root/f'{label}_step4.pt')
    if max(initial_losses)-min(initial_losses)>1e-10:raise ValueError('initial common-noise losses differ')
    write_json(root/'summary.json',dict(complete=True,traces=traces,parameters=parameters,
        initial_common_noise_loss_exact=True,subset_max_difference=subset_error,total_seconds=time.monotonic()-start,
        protocol_sha256=sha(root/'protocol.json'),role='engineering only; no development evaluation',
        strict_oof=False,confirmation_C_D_read=False))
    print('Complete actual4-update inducing-flow spike',flush=True)


if __name__=='__main__':main()
