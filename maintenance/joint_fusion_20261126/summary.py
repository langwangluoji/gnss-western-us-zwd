"""Paired fusion evaluation and independent-pair center preservation; no fitting."""
import sys
from pathlib import Path
import numpy as np
import torch
from campaign import ROOT,ENGINE,LEGACY,RESULTS,OUTPUT,HERE,LOCK,read,paths,validate_lock,verify_training,call
from fusion_contract import centered_pairs,require_equal_receipt
sys.path.append(str(LEGACY))
from weather_controls_summary import monthly_record,event_records,fair_mean_mse
from mean_hs_summary import safe_comparison
from inducing_calibration_audit import rank_coverage
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_model_seal import evaluation_marker
from tools.terrain_diffusion.native_weather_evaluation import calendar_shape,point_metrics,fair_crps


def pair_output(folder,cache,source,samples,mean,truth):
    destination=folder.with_name('paired_'+folder.name)
    expected=dict(source_evaluation_sha256=source['prediction_sha256'],lock_sha256=sha(LOCK),
        cache_manifest_sha256=sha(cache/'manifest.json'),members=16,independent_iid_member_noise=True,
        mechanism='mu_joint + (independent even member - independent odd member)/sqrt(2)',
        postprocessor_sha256=sha(__file__),strict_oof=False,confirmation_C_D_read=False)
    if destination.exists():
        evaluation_marker(destination);require_equal_receipt(read(destination/'lineage.json'),expected)
    else:
        x=centered_pairs(samples,mean).astype('float32');crps=fair_crps(x,truth)
        destination.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(destination/'evaluation.npz',prediction=mean,truth=truth,samples=x,
            crps=crps.astype('float32'),distribution_mean=mean)
        write_json(destination/'metrics.json',dict(deterministic_mean=point_metrics(mean,truth),
            sample_mean=point_metrics(x.mean(0),truth),fair_crps_mm_by_horizon=np.nanmean(crps,(0,1,2,3)).tolist(),
            forecast_increment_fair_crps_mm=float(np.nanmean(fair_crps(np.diff(x,axis=-1),np.diff(truth,axis=-1)))),
            negative_sample_fraction=float((x<0).mean()),analytic_distribution_mean=point_metrics(mean,truth)))
        write_json(destination/'lineage.json',{**expected,'prediction_sha256':sha(destination/'evaluation.npz')})
        del x,crps
    if not (destination/'joint_metrics.json').exists():
        call('probability_joint_audit.py',['--cache',cache,'--evaluation',destination,'--out',destination],
            'post_'+destination.parent.name+'_'+destination.name,helper=True)
    return destination


def collect(folder,cache,seed,day,name,paired=False):
    evaluation_marker(folder);lin=read(folder/'lineage.json');metrics=read(folder/'metrics.json')
    joint=read(folder/'joint_metrics.json')
    if joint['source_lineage_sha256']!=sha(folder/'lineage.json') or joint['code_sha256']!=sha(LEGACY/'probability_joint_audit.py'):
        raise ValueError('joint scoring source differs')
    _,embedding,_=paths(seed,day)
    geometry=np.asarray(read(cache/'geometry.json')['dev_static'])
    nearest=calendar_shape(np.load(cache/'nearest.npy'),12)
    with np.load(folder/'evaluation.npz') as z:
        x=z['samples'];y=z['truth'];mu=z['prediction'];crps=z['crps'];K=16 if paired else 32
        if x.shape!=(K,3,12,72,137,7) or not np.isfinite(x).all():raise ValueError('incomplete fusion ensemble')
        np.testing.assert_array_equal(mu,calendar_shape(np.load(embedding/'mean.npy'),12))
        np.testing.assert_array_equal(y,calendar_shape(np.load(cache/'truth.npy'),12))
        if paired:np.testing.assert_array_equal(z['distribution_mean'],mu)
        np.testing.assert_allclose(crps,fair_crps(x,y),atol=2e-5,rtol=1e-6,equal_nan=True)
        records={f'crps_h{h}':monthly_record(crps[...,h],1) for h in range(7)}
        records['increment']=monthly_record(fair_crps(np.diff(x,axis=-1),np.diff(y,axis=-1)),1)
        np.testing.assert_allclose(records['increment']['value'],metrics['forecast_increment_fair_crps_mm'],atol=1e-10,rtol=0)
        records.update(event_records(x,y,read(LEGACY/'event_audit_lock.json')))
        for h in (0,6):
            records[f'fair_mean_mse_h{h}']=monthly_record(fair_mean_mse(x[...,h],y[...,h]),1)
            records[f'base_mse_h{h}']=monthly_record((mu[...,h].astype('float64')-y[...,h])**2,1)
        if K==32:
            coverage=rank_coverage(x,y)
            for h in (0,6):records[f'randomized80_h{h}']=monthly_record(coverage['randomized80'][...,h],1)
        for distance,table in joint['spatial_pair_crps'].items():
            for h in ('h0','h6'):records[f'pair_{distance}_{h}']={**table[h],'value':table[h]['fair_crps_mm']}
        records['energy']=dict(value=joint['energy']['mean_mm'],monthly_score_sum=joint['energy']['monthly_mean_mm'],monthly_valid_count=[1]*12)
        masks={}
        for lo,hi in ((0,50),(50,100),(100,150),(150,300),(300,np.inf)):
            masks[f'distance_{lo}-{hi:g}']=(nearest>=lo)&(nearest<hi)
        for feature,v,edges in [('height',geometry[:,2]*3000,[-np.inf,500,1500,np.inf]),('coast',geometry[:,8]*500,[0,50,150,300,np.inf]),('roughness_code',geometry[:,5],[0,.5,1.5,np.inf])]:
            for lo,hi in zip(edges[:-1],edges[1:]):masks[f'{feature}_{lo:g}-{hi:g}']=np.broadcast_to(((v>=lo)&(v<hi))[None,None,None],y.shape[:-1])
        for season,months in [('DJF',[11,0,1]),('MAM',[2,3,4]),('JJA',[5,6,7]),('SON',[8,9,10])]:
            mask=np.zeros(y.shape[:-1],bool);mask[:,months]=True;masks['season_'+season]=mask
        for label,mask in masks.items():
            for h in (0,6):
                v=np.where(mask,crps[...,h],np.nan)
                if np.isfinite(v).any():records[f'{label}_crps_h{h}']=monthly_record(v,1)
        valid=np.isfinite(y);q=np.quantile(x,[.1,.9],axis=0)
        empirical=(((y>=q[0])&(y<=q[1])&valid).sum((0,1,2,3))/valid.sum((0,1,2,3))).tolist()
        row=dict(seed=seed,day=day,name=name,members=K,records=records,
            point_R_F6_mm=metrics['deterministic_mean']['horizon_macro_rmse_mm'][::6],
            sample_mean_R_F6_mm=metrics['sample_mean']['horizon_macro_rmse_mm'][::6],
            probability_analytic_mean_R_F6_mm=metrics.get('analytic_distribution_mean',metrics.get('gaussian_distribution_mean',{})).get('horizon_macro_rmse_mm',[None]*7)[::6],
            empirical10_90_coverage=empirical,negative_fraction_by_horizon=(x<0).mean((0,1,2,3,4)).tolist(),
            source={n:sha(folder/n) for n in ('evaluation.npz','lineage.json','metrics.json','joint_metrics.json')})
    print('collected',day,seed,name,records['crps_h6']['value'],flush=True)
    return row


def main():
    torch.set_num_threads(2);validate_lock();rows=[]
    for seed in (42,17):
        validate_lock(seed)
        for kind in ('gaussian','gp','affine','local'):
            verify_training(OUTPUT/f's{seed}'/dict(gaussian='S175',gp='G265').get(kind,kind),seed,kind)
        for day in (8,15):
            cache,_,_=paths(seed,day)
            for mode in ('G265','affine','local'):
                folder=OUTPUT/f'calendar{day}/{mode}_s{seed}';source=read(folder/'lineage.json')
                # Native marker validates the saved model seal during campaign;
                # retain its probability source and calendar receipt here as well.
                gaussian=OUTPUT/f's{seed}/S175';gp=OUTPUT/f's{seed}/G265'
                _,emb,_=paths(seed,day)
                require_equal_receipt(source,dict(cache_manifest_sha256=sha(cache/'manifest.json'),embedding_manifest_sha256=sha(emb/'manifest.json'),members=32,seed=261126,independent_iid_member_noise=True))
                src=source['source']
                require_equal_receipt(src,dict(gaussian_weights_sha256=sha(gaussian/'last.pt'),gaussian_protocol_sha256=sha(gaussian/'protocol.json'),
                    copula_sha256=sha(gaussian/'copula.json'),mean_weights_sha256=read(emb/'manifest.json')['mean_weights_sha256']))
                from tools.terrain_diffusion.native_weather_gp_experiment import codes as gp_codes
                from tools.terrain_diffusion.native_weather_gp_diffusion_experiment import codes as diffusion_codes
                if source['code_sha256']!=(gp_codes() if mode=='G265' else diffusion_codes()):raise ValueError('fusion evaluation implementation differs')
                model=gp if mode=='G265' else OUTPUT/f's{seed}/{mode}'
                require_equal_receipt(source,dict(probability_weights_sha256=sha(model/'last.pt')))
                if mode!='G265':
                    require_equal_receipt(source,dict(solver='heun0',sampling_steps=20))
                    require_equal_receipt(src,dict(weather_gp_protocol_sha256=sha(gp/'protocol.json'),weather_gp_weights_sha256=sha(gp/'last.pt'),weather_gp_seal_sha256=sha(gp/'model_artifact_seal.json')))
                with np.load(folder/'evaluation.npz') as z:
                    paired=pair_output(folder,cache,source,z['samples'],z['prediction'],z['truth'])
                rows.append(collect(folder,cache,seed,day,mode))
                rows.append(collect(paired,cache,seed,day,'paired_'+mode,paired=True))
    comparisons={};lookup={(r['day'],r['seed'],r['name']):r for r in rows}
    for day in (8,15):
        for a,b in [('local','affine'),('local','G265'),('affine','G265'),('paired_local','paired_affine'),('paired_local','paired_G265'),('paired_affine','paired_G265')]:
            keys=set.intersection(*(set(lookup[day,s,n]['records']) for s in (42,17) for n in (a,b)))
            comparisons[f'day{day}/{a}_vs_{b}']={k:safe_comparison([(lookup[day,s,a]['records'][k],lookup[day,s,b]['records'][k]) for s in (42,17)]) for k in sorted(keys)}
    gaps={}
    for day in (8,15):
        for name in ('G265','affine','local','paired_G265','paired_affine','paired_local'):
            gaps[f'day{day}/{name}']={f'h{h}':safe_comparison([(lookup[day,s,name]['records'][f'fair_mean_mse_h{h}'],lookup[day,s,name]['records'][f'base_mse_h{h}']) for s in (42,17)]) for h in (0,6)}
    out=OUTPUT/'paired';out.mkdir(parents=True,exist_ok=True)
    payload=dict(rows=rows,comparisons=comparisons,fair_conditional_mean_MSE_gap_mm2=gaps,lock_sha256=sha(LOCK),
        summary_code_sha256=sha(__file__),all_point_fields_equal_frozen_joint=True,strict_oof=False,confirmation_C_D_read=False,
        interval='2000 common-month resamples; equal fixed seeds; two exposed2020 calendars separately; no multiplicity adjustment',
        raw_members=32,center_preserved_members=16,point_prediction='frozen joint v2, not ensemble mean')
    write_json(out/'summary.json',payload);render(out,rows,comparisons,gaps)


def render(out,rows,comparisons,gaps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'svg.fonttype':'path'})
    value=lambda day,name,key:np.mean([r['records'][key]['value'] for r in rows if r['day']==day and r['name']==name])
    lines=['# 最佳joint v2基础场＋天气残差扩散：正式配对结果','',
        '2016–2019连续训练，两种子每臂3000块；2020主/补日历均为开发。点输出逐值等于原joint v2；raw K32与独立不重叠双场差分K16分别比较。概率Gaussian残差中心与确定性点输出分开。','',
        '| 日历 | 模型 | 当前CRPS/mm | F6 CRPS/mm | 增量CRPS/mm | 近邻F6差值CRPS/mm | energy/mm |',
        '|---|---|---:|---:|---:|---:|---:|']
    names=('G265','affine','local','paired_G265','paired_affine','paired_local')
    for day in (8,15):
        for name in names:lines.append('| '+str(day)+' | '+name+' | '+' | '.join(f'{value(day,name,k):.6f}' for k in ('crps_h0','crps_h6','increment','pair_0-50_h6','energy'))+' |')
    lines+=['','## 主比较及均值保持比较','']
    for day in (8,15):
        for a,b in [('local','affine'),('paired_local','paired_affine')]:
            r=comparisons[f'day{day}/{a}_vs_{b}']['crps_h6']
            lines.append(f"- 日历{day}，{a}−{b}：F6差值{r['delta']:.6f}mm，改善{r['gain_percent']:.4f}%，95%月份区间{r['delta_ci95']}。")
    lines+=['','## 点输出与采样均值','', '| 日历 | 模型 | 确定性R / F6 mm | K成员均值R / F6 mm | 当前均值MSE差/mm²及区间 |','|---|---|---|---|---|']
    for day in (8,15):
        for name in names:
            rs=[r for r in rows if r['day']==day and r['name']==name];p=np.mean([r['point_R_F6_mm'] for r in rs],0);q=np.mean([r['sample_mean_R_F6_mm'] for r in rs],0);g=gaps[f'day{day}/{name}']['h0']
            lines.append(f"| {day} | {name} | {p[0]:.6f} / {p[1]:.6f} | {q[0]:.6f} / {q[1]:.6f} | {g['delta']:.6f} {g['delta_ci95']} |")
    lines+=['','中心保持场的总体均值严格等于joint（float32存盘有舍入），K16实际成员均值仍含采样方差；均值MSE扣除s²/K是Monte Carlo估计，不能当作新的确定性RMSE。两个日历不拼接增量。','',
        '完整事件、条件站距离/海岸/高程/地形/季节、有限成员覆盖和来源见[summary.json](summary.json)。季节区间出现空月份抽样时标记NA。strict_oof=false；旧候选池未来可用性与历史质控限制保留，确认及原C/D未读。','',
        '![融合概率收益](fusion_effects.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,(key,title) in zip(axes,[('crps_h6','F6 CRPS差值 / mm'),('increment','小时增量CRPS差值 / mm'),('pair_0-50_h6','近邻F6差值CRPS差值 / mm')]):
        labels=[]
        for i,(day,prefix) in enumerate([(8,''),(15,''),(8,'paired_'),(15,'paired_')]):
            r=comparisons[f'day{day}/{prefix}local_vs_{prefix}affine'][key];v=r['delta'];lo,hi=r['delta_ci95']
            ax.errorbar(v,3-i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',capsize=4,color='#168878' if not prefix else '#8d69ad');labels.append(f'{day}日历 '+('K16均值保持' if prefix else 'K32原始'))
        ax.axvline(0,color='#64748b',lw=1);ax.set(title=title,yticks=range(4),yticklabels=labels[::-1]);ax.grid(alpha=.2)
    fig.suptitle('Local扩散 − 匹配Affine对照｜冻结最佳joint v2均值；负值表示误差减少')
    fig.savefig(out/'fusion_effects.png',dpi=170);fig.savefig(out/'fusion_effects.svg');plt.close(fig)


if __name__=='__main__':main()
