"""Label-free probe of learned latent and decoded nonlinear variance, not a new forecast model."""
from pathlib import Path
from types import SimpleNamespace
import torch
from campaign import RESULTS,read
from inducing_flow_series import source_paths,ARMS
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_inducing_flow_experiment import codes,batch
from tools.terrain_diffusion.native_inducing_flow import InducingLatentFlow
from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import WeatherSources
from tools.terrain_diffusion.native_model_seal import validate_model_seal


@torch.no_grad()
def main():
    torch.set_num_threads(2);root=RESULTS/'20261126_inducing_latent_flow';rows=[];sources={}
    for seed in (42,17):
        source=WeatherSources(SimpleNamespace(seed=seed,device='cuda',smoke_cache=False,**source_paths(seed)))
        if source.data.development:raise ValueError('probe uses training histories only')
        for family,domain in ARMS:
            name=f'{family}_{domain}_s{seed}';folder=root/name
            seal=validate_model_seal(folder,'COMPLETE_INDUCING_FLOW',codes());protocol=read(folder/'protocol.json')
            model=InducingLatentFlow(family,domain).cuda().eval();model.load_state_dict(torch.load(folder/'last.pt',map_location='cuda',weights_only=True))
            anchors=torch.tensor(protocol['anchors'],device='cuda',dtype=torch.float64)
            for block in (0,1000,2000,2800):
                b,_,_,residual,_,_,_=batch(source,block,anchors,2,seed,True)
                c=b['condition'][:1];global_c=b['global_condition'][:1];basis=b['basis'][:1].double();sigma=b['std'][:1].flatten(1).double()
                scale,phi=model.query(c);B,Q,H,D=phi.shape
                P=(phi[...,None]*torch.eye(H,device='cuda')[None,None,:,None,:]).reshape(B,Q*H,D*H).double()
                A=scale.flatten(1).double();M=(A[...,None]*basis+P)*sigma[...,None]
                torch.manual_seed(seed+261126+block)
                fit=torch.randn(1024,1,D,H,device='cuda');test=torch.randn(1024,1,D,H,device='cuda')
                fit_g=model.latent(fit,global_c).flatten(2)[:,0].double()
                x=fit.flatten(2)[:,0].double();F=torch.linalg.lstsq(x,fit_g).solution
                test_g=model.latent(test,global_c).flatten(2)[:,0].double();eta=test.flatten(2)[:,0].double()
                residual_g=test_g-eta@F
                decoded=residual_g@M[0].T;total_g=test_g@M[0].T
                remaining=((A*sigma).square()*residual[:1].diagonal(dim1=-2,dim2=-1)).sum()
                latent_fraction=float(residual_g.square().mean()/test_g.square().mean())
                field_fraction=float(decoded.square().sum(-1).mean()/(remaining+total_g.square().sum(-1).mean()))
                if family=='affine' and (latent_fraction>1e-10 or field_fraction>1e-10):raise ValueError('linear control fails surrogate reconstruction')
                rows.append(dict(seed=seed,name=name,training_block_index0=block,latent_nonlinear_variance_fraction=latent_fraction,
                    decoded_field_nonlinear_variance_fraction=field_fraction,latent_departure_from_identity=float((test_g-eta).square().mean()/eta.square().mean()),
                    inducing_original_variance_fraction=float(basis.square().sum(-1).mean()),
                    query_scale_min=float(scale.min()),query_scale_max=float(scale.max()),phi_rms=float(phi.square().mean().sqrt())))
            sources[name]=dict(weights_sha256=seal['weights_sha256'],protocol_sha256=seal['protocol_sha256'])
    out=root/'nonlinearity_probe';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',dict(rows=rows,sources=sources,code_sha256=sha(__file__),fit_noise=1024,independent_test_noise=1024,
        condition='first origin in four fixed training blocks per seed; no GNSS target values used in surrogate fit',
        interpretation='out-of-noise-sample linear-surrogate residual energy; diagnostic, not exact cumulants or forecast skill',
        full_Gaussian_residual_retained=True,confirmation_C_D_read=False))
    lines=['# 已训练潜流的非线性是否进入最终场','',
        '每种子选固定四个训练历史的首起点，不读GNSS目标值进行拟合。用1024个Gaussian噪声拟合latent线性代理，再用独立1024噪声检验残差能量。场中分母包括完整剩余GP方差，分子只计算非线性代理残差经实际解码后的能量；它是受所选条件和Monte Carlo影响的诊断，不是精确高阶累积量或预测收益。','',
        '| 模型 | latent非线性方差比例 | 最终场非线性方差比例 | 原诱导投影方差比例 |','|---|---:|---:|---:|']
    for family,domain in ARMS:
        selected=[r for r in rows if r['name'].startswith(f'{family}_{domain}_s')]
        average=lambda key:sum(r[key] for r in selected)/len(selected)
        lines.append(f'| {family}_{domain} | {average("latent_nonlinear_variance_fraction"):.6g} | {average("decoded_field_nonlinear_variance_fraction"):.6g} | {average("inducing_original_variance_fraction"):.6g} |')
    lines+=['','完整32个条件点和权重来源见[summary.json](summary.json)。Affine近零残差用于核验诊断方法；odd可以保持中心对称而具有非线性。结果不用于选取新开发日期或改变本轮主指标。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Complete label-free learned nonlinearity probe',flush=True)


if __name__=='__main__':main()
