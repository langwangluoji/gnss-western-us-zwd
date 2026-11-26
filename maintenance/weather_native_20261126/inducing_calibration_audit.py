"""Finite-K rank calibration and physical-support diagnostics; no model fitting."""
import argparse
from pathlib import Path
import numpy as np


def rank_coverage(samples,truth):
    if len(samples)!=32:raise ValueError('registered calibration uses32 IID members')
    ordered=np.sort(samples,axis=0);valid=np.isfinite(truth)
    a=np.where(valid,(truth>=ordered[2])&(truth<=ordered[29]),np.nan)
    b=np.where(valid,(truth>=ordered[3])&(truth<=ordered[28]),np.nan)
    return dict(rank3_30=a,rank4_29=b,randomized80=.7*a+.3*b)


def main():
    from campaign import RESULTS,read
    from inducing_flow_series import ARMS,LOCK
    from inducing_flow_summary import require_pairing
    from tools.terrain_diffusion.data import sha,write_json
    from tools.terrain_diffusion.native_model_seal import evaluation_marker,validate_model_seal
    from tools.terrain_diffusion.native_inducing_flow_experiment import codes
    from weather_controls_summary import monthly_record
    p=argparse.ArgumentParser();p.add_argument('--baseline-only',action='store_true');p.add_argument('--calendar-day',type=int,choices=[8,15],default=8)
    args=p.parse_args();lock=read(LOCK);root=RESULTS/'20261126_inducing_latent_flow';rows=[]
    if lock['code_sha256']!=codes():raise ValueError('registered model implementation differs')
    for seed in (42,17):
        for family,domain in [('gaussian','time'),*([] if args.baseline_only else ARMS)]:
            name=family+'_'+domain;folder=root/f'calendar{args.calendar_day}/{name}_s{seed}'
            seal=None if family=='gaussian' else validate_model_seal(root/f'{name}_s{seed}','COMPLETE_INDUCING_FLOW',codes())
            evaluation_marker(folder,seal);lin=read(folder/'lineage.json');require_pairing(lin,family,domain,seed,lock['sample_seed'],lock['evaluation_device'])
            calendar=lock['sources'][str(seed)]['calendars'][str(args.calendar_day)]
            if lin['cache_manifest_sha256']!=calendar['cache_manifest_sha256'] or lin['embedding_manifest_sha256']!=calendar['embedding_manifest_sha256']:
                raise ValueError('registered calendar differs')
            with np.load(folder/'evaluation.npz') as data:
                x=data['samples'];y=data['truth'];base=data['prediction'];valid=np.isfinite(y)
                coverage=rank_coverage(x,y)
                records={f'{name}_h{h}':monthly_record(value[...,h],1) for name,value in coverage.items() for h in (0,6)}
                if 'gaussian_distribution_std' in data:
                    c=np.where(valid,abs(y-base)<=1.2815515655446004*data['gaussian_distribution_std'],np.nan)
                    records.update({f'analytic80_h{h}':monthly_record(c[...,h],1) for h in (0,6)})
                rank=(x<y[None]).sum(0);histograms={str(h):np.bincount(rank[...,h][valid[...,h]],minlength=33).tolist() for h in (0,6)}
                sample_mean=np.mean(x,axis=0,dtype='float64');variance=np.var(x,axis=0,dtype='float64',ddof=1)
                fair_mean_mse=(sample_mean-y)**2-variance/len(x)
                mean_gap=np.where(valid,fair_mean_mse-(base-y)**2,np.nan)
                records.update({f'finite_MC_mean_MSE_gap_h{h}':monthly_record(mean_gap[...,h],1) for h in (0,6)})
                rows.append(dict(seed=seed,name=name,records=records,rank_histograms=histograms,
                    negative_sample_fraction_by_horizon=(x<0).mean((0,1,2,3,4)).tolist(),
                    tied_observation_cells_R_F6=[int(((x[...,h]==y[None,...,h]).any(0)&valid[...,h]).sum()) for h in (0,6)],
                    source_lineage_sha256=sha(folder/'lineage.json'),source_evaluation_sha256=lin['prediction_sha256']))
            print(name,seed,{k:v['value'] for k,v in records.items() if k.endswith('h6')},flush=True)
    out=root/('baseline_calibration' if args.baseline_only else f'calibration_calendar{args.calendar_day}');out.mkdir(parents=True,exist_ok=True)
    write_json(out/'summary.json',dict(rows=rows,code_sha256=sha(__file__),lock_sha256=sha(LOCK),
        rank3_30_exchangeable_reference=27/33,rank4_29_exchangeable_reference=25/33,
        randomized80_reference=.8,randomized80='expected coverage under independent interval randomization: .7 rank3_30 + .3 rank4_29',
        quantile10_90_coverage='descriptive finite-member empirical interval; not exactly80 percent under calibration',
        mean_gap='Monte Carlo diagnostic only; all candidate analytic point means equal frozen TF3000',
        strict_oof=False,confirmation_C_D_read=False,phase='baseline-only' if args.baseline_only else 'all sealed models'))
    lines=['# 有限成员覆盖率与物理支撑诊断','',
        'K32 IID条件下，连续且交换的真值/成员总体，rank3–30参考覆盖率27/33=81.818%，rank4–29为25/33=75.758%。以0.7/0.3独立随机选择两区间，其期望覆盖率为80%；表内报告该随机化的期望，不对随机硬区间另抽一次噪声。原经验10%–90%覆盖率保留为描述性量。','',
        '| 模型 | 种子 | 当前随机化覆盖率 | F6随机化覆盖率 | F6解析Gaussian覆盖率 | F6负样本比例 |','|---|---|---:|---:|---:|---:|']
    for r in rows:
        record=r['records'];analytic=record.get('analytic80_h6',{}).get('value')
        lines.append(f"| {r['name']} | {r['seed']} | {record['randomized80_h0']['value']:.5f} | {record['randomized80_h6']['value']:.5f} | {analytic if analytic is not None else 'NA'} | {r['negative_sample_fraction_by_horizon'][6]:.6f} |")
    lines+=['','负值比例的分母为所有生成查询，包括无标签站；覆盖率与秩图只用有效标签。标签量化导致的相等情况单列；交换参考以连续分布为条件。输出不作裁剪，裁剪会改变均值、尺度和概率评分，需另设模型合同。','',
        '月份记录、秩直方图与有限成员均值MSE诊断见[summary.json](summary.json)。本诊断不修改训练、主指标或选模规则，2020日期已开发暴露。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
