"""Final source-checked weather-scale versus matched-covariance tail comparisons."""
from pathlib import Path
import numpy as np
import torch
from campaign import RESULTS,read
from weather_calibration_run import ROOT,ARMS,VARIANTS,codes,cached,model_load,constants,features,GAMMA_SEED
from weather_calibration_lock import LOCK,validate_lock,require_lineage
from tools.terrain_diffusion.data import sha,write_json
from tools.terrain_diffusion.native_model_seal import evaluation_marker
from tools.terrain_diffusion.native_weather_evaluation import fair_crps,calendar_shape,point_metrics
from tools.terrain_diffusion.native_weather_calibration import transform_ensemble
from weather_controls_summary import monthly_record,event_records
from mean_hs_summary import safe_comparison
from inducing_calibration_audit import rank_coverage


def verify_models(seed,stats):
    result={}
    for arm in ARMS:
        _,seal=model_load(seed,arm);folder=ROOT/f'{arm}_s{seed}';status=read(folder/'status.json');p=read(folder/'protocol.json')
        state=torch.load(folder/'checkpoint_3000.pt',map_location='cpu',weights_only=True);n=stats['positive_blocks']
        if p['parameters']!=10049 or status['supervised_updates']!=n or status['finite_target_cells']!=stats['finite_target_cells']:
            raise ValueError('parameter/actual supervised budget differs')
        opt=state['optimizer'];expected={v for g in opt['param_groups'] for v in g['params']}
        if state['step']!=3000 or not expected or set(opt['state'])!=expected or any(int(v['step'])!=n for v in opt['state'].values()) or state['scheduler']['last_epoch']!=n:
            raise ValueError('optimizer/scheduler does not cover registered supervised clock')
        result[arm]=seal
    return result


def main():
    torch.set_num_threads(2)
    lock=validate_lock();rows=[];records={};sources={};event_lock=read(Path(__file__).with_name('event_audit_lock.json'))
    for seed in (42,17):
        train,a,stats=cached(seed);dev,da,_=cached(seed,8);seals=verify_models(seed,stats);const=constants(seed)
        x=features(da,stats);g,_=model_load(seed,'G_weather');extra,_=model_load(seed,'G_extra');t,_=model_load(seed,'T_weather')
        with torch.no_grad():
            base=g(x.reshape(-1,90)).reshape(216,12).numpy();more=extra(x.reshape(-1,90)).reshape(216,12).numpy();nu=t(x.reshape(-1,90)).reshape(216,12).numpy()
        parent=RESULTS/f'20261126_inducing_latent_flow/calendar8/affine_time_s{seed}'
        with np.load(parent/'evaluation.npz') as z:mu0=z['prediction'];truth0=z['truth'];x0=z['samples']
        base_std=None;base_s2=None
        for name in VARIANTS:
            folder=ROOT/f'calendar8/{name}_s{seed}';evaluation_marker(folder);lin=read(folder/'lineage.json');m=read(folder/'metrics.json');joint=read(folder/'joint_metrics.json')
            primary=seals.get(name,seals['G_weather'])
            expected=dict(name=name,seed=seed,members=32,code_sha256=codes(),final_lock_sha256=sha(LOCK),
                source_evaluation_sha256=lock['sources'][str(seed)]['parent_evaluation_sha256'],
                training_statistic_manifest_sha256=sha(train/'manifest.json'),development_statistic_manifest_sha256=sha(dev/'manifest.json'),
                constants_sha256=sha(ROOT/f'constants_s{seed}.json'),independent_iid_member_noise=True,
                process_protocol_sha256=primary['protocol_sha256'],probability_weights_sha256=primary['weights_sha256'],
                G_weather_weights_sha256=seals['G_weather']['weights_sha256'],G_extra_weights_sha256=seals['G_extra']['weights_sha256'],T_weather_weights_sha256=seals['T_weather']['weights_sha256'],
                gamma_seed=GAMMA_SEED if name=='T_weather' or (name=='T_constant' and const['nu'] is not None) else None)
            require_lineage(lin,expected)
            if joint['source_lineage_sha256']!=sha(folder/'lineage.json') or joint['code_sha256']!=lock['helpers']['probability_joint_audit.py']:
                raise ValueError('joint score source differs')
            with np.load(folder/'evaluation.npz') as z:
                y=z['truth'];mu=z['prediction'];samples=z['samples'];sc=z['variance_scale2'];std=z['distribution_std']
                np.testing.assert_array_equal(mu,mu0);np.testing.assert_array_equal(y,truth0)
                if samples.shape!=(32,3,12,72,137,7) or not np.isfinite(samples).all():raise ValueError('incomplete/nonfinite ensemble')
                s2=np.ones_like(base) if name=='fixed' else np.full_like(base,const['constant_scale2']) if name=='constant' else base*more if name=='G_extra' else base
                v=nu if name=='T_weather' else np.full_like(nu,const['nu']) if name=='T_constant' and const['nu'] is not None else None
                np.testing.assert_array_equal(sc,calendar_shape(s2,12))
                if v is not None:np.testing.assert_array_equal(z['nu'],calendar_shape(v,12))
                if name=='G_weather':base_std=std.copy();base_s2=sc.copy()
                if name in ('T_weather','T_constant'):
                    np.testing.assert_array_equal(sc,base_s2);np.testing.assert_array_equal(std,base_std)
                if name=='fixed':np.testing.assert_array_equal(samples,x0)
                ids=np.array([0,68,136]);expected_x=transform_ensemble(torch.from_numpy(x0[...,ids,:].reshape(32,-1,3,7)),
                    torch.from_numpy(mu0[...,ids,:].reshape(-1,3,7)),torch.from_numpy(sc.reshape(-1)),
                    None if v is None else torch.from_numpy(calendar_shape(v,12).reshape(-1)),GAMMA_SEED).numpy().reshape(32,3,12,72,3,7)
                np.testing.assert_array_equal(samples[...,ids,:],expected_x)
                crps=fair_crps(samples,y);np.testing.assert_allclose(z['crps'],crps,atol=2e-5,rtol=1e-6,equal_nan=True)
                r={f'crps_h{h}':monthly_record(crps[...,h],1) for h in (0,6)}
                r['increment']=monthly_record(fair_crps(np.diff(samples,axis=-1),np.diff(y,axis=-1)),1)
                r['joint_nll_per_dimension']=monthly_record(z['nll_per_dimension'],1)
                r.update(event_records(samples,y,event_lock));coverage=rank_coverage(samples,y)
                for h in (0,6):r[f'randomized80_h{h}']=monthly_record(coverage['randomized80'][...,h],1)
                for distance,table in joint['spatial_pair_crps'].items():
                    for h in ('h0','h6'):r[f'pair_{distance}_{h}']={**table[h],'value':table[h]['fair_crps_mm']}
                r['energy']=dict(value=joint['energy']['mean_mm'],monthly_score_sum=joint['energy']['monthly_mean_mm'],monthly_valid_count=[1]*12)
                np.testing.assert_allclose(r['increment']['value'],m['forecast_increment_fair_crps_mm'],atol=1e-10,rtol=0)
                np.testing.assert_allclose(r['joint_nll_per_dimension']['value'],m['joint_nll_per_dimension'],atol=1e-12,rtol=0)
                point=point_metrics(mu,y)
            records[seed,name]=r;rows.append(dict(seed=seed,name=name,records=r,point_R_F6_rmse_mm=point['horizon_macro_rmse_mm'][::6],
                negative_sample_fraction_by_horizon=m['negative_sample_fraction_by_horizon'],covariance_matched_to_G_weather=name in ('T_weather','T_constant')))
            sources[f'{seed}/{name}']=dict(lineage_sha256=sha(folder/'lineage.json'),metrics_sha256=sha(folder/'metrics.json'),joint_sha256=sha(folder/'joint_metrics.json'),evaluation_sha256=lin['prediction_sha256'])
            print(seed,name,r['crps_h6']['value'],r['joint_nll_per_dimension']['value'],flush=True)
            del samples,crps,coverage,y,mu
        del x0,mu0,truth0
    comparisons={};edges=[(n,'fixed') for n in VARIANTS if n!='fixed']+[('T_weather','G_weather'),('T_weather','G_extra'),('T_weather','T_constant'),('G_extra','G_weather')]
    for a,b in edges:
        keys=set.intersection(*(set(records[s,n]) for s in (42,17) for n in (a,b)))
        comparisons[f'{a}_vs_{b}']={k:safe_comparison([(records[s,a][k],records[s,b][k]) for s in (42,17)]) for k in sorted(keys)}
    out=ROOT/'paired';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',dict(rows=rows,comparisons=comparisons,sources=sources,primary=lock['primary'],lock_sha256=sha(LOCK),
        code_sha256=sha(__file__),matched_covariance_arrays_exact=True,all_centers_truth_exact=True,
        interval='2000 common month resamples, equal fixed seeds; exposed sampled2020, no multiplicity correction',
        strict_oof=False,confirmation_C_D_read=False,user_end_condition='stop new experiments after this final comparison'))
    render(out,rows,comparisons)


def render(out,rows,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    value=lambda n,k:np.mean([r['records'][k]['value'] for r in rows if r['name']==n])
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,(key,title) in zip(axes,[('crps_h6','F6 CRPS difference (mm)'),('joint_nll_per_dimension','Joint NLL/d difference'),('increment','Increment CRPS difference (mm)')]):
        for i,edge in enumerate(['G_weather_vs_fixed','T_weather_vs_G_weather','T_weather_vs_G_extra']):
            r=comparisons[edge][key];v=r['delta'];lo,hi=r['delta_ci95'];ax.errorbar(v,2-i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4)
        ax.axvline(0,color='#64748b',lw=1);ax.set(yticks=[2,1,0],yticklabels=['Weather G - fixed','Weather t - matched G','Weather t - extra G'],title=title);ax.grid(alpha=.2);ax.tick_params(axis='y',labelsize=7)
    fig.savefig(out/'weather_scale_tail.png',dpi=170);plt.close(fig)
    lines=['# 最后一轮天气尺度与同协方差Student-t对照','',
        '原2016–2019连续3000块，两个固定种子；2020每月8–10日三布局、137查询、K32 IID开发评价。均值全部等于原TF3000；G_weather/T_weather尺度和标准差数组逐值相同，新增MLP均10049参数，额外Gaussian与尾部阶段预算相同。','',
        '| 模型 | 当前CRPS/mm | F6 CRPS/mm | 内部增量CRPS/mm | 近邻F6差值CRPS/mm | joint NLL/d | 80%参考F6覆盖率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n in VARIANTS:lines.append('| '+n+' | '+' | '.join(f'{value(n,k):.6f}' for k in ('crps_h0','crps_h6','increment','pair_0-50_h6','joint_nll_per_dimension','randomized80_h6'))+' |')
    lines+=['','## 预注册主比较及额外预算控制','']
    for edge in ['T_weather_vs_G_weather','T_weather_vs_G_extra','T_weather_vs_T_constant']:
        r=comparisons[edge]['crps_h6'];lines.append(f"- {edge}：F6差值{r['delta']:.6f}mm，95%共同月份区间{r['delta_ci95']}。")
    lines+=['','![六小时与联合评分](weather_scale_tail.png)','','完整事件、覆盖率、源和区间见[summary.json](summary.json)。联合对数评分与单站精度分开解释；strict_oof=false，既有未来可用性选站与历史质控限制保留，确认及原C/D未读。本轮结束后按用户要求停止新增实验。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
