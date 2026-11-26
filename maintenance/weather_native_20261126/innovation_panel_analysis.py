"""Derived R2 and residual energy scores; the frozen probe fit is unchanged."""
import numpy as np


def relative_scores(candidate_mse,reference_mse,reference_bias):
    variance=np.asarray(reference_mse)-np.asarray(reference_bias)**2
    if np.any(variance < -1e-8):raise ValueError('reference second moment smaller than squared mean')
    r2=1-np.divide(candidate_mse,variance,out=np.full_like(variance,np.nan,dtype='float64'),where=variance>1e-12)
    energy=1-np.divide(candidate_mse,reference_mse,out=np.full_like(variance,np.nan,dtype='float64'),where=np.asarray(reference_mse)>1e-12)
    return dict(predictive_r2=r2,relative_residual_energy_reduction=energy,reference_residual_variance_mm2=np.maximum(variance,0))


def main():
    from campaign import RESULTS,read
    from tools.terrain_diffusion.data import sha,write_json
    from paired_mean_summary import plain
    out=RESULTS/'20261126_leaveout_innovation_panel/paired';source=read(out/'summary.json')
    reference=next(r for r in source['rows'] if r['name']=='none');rows=[]
    for row in source['rows']:
        per_seed=[]
        for i,seed in enumerate((42,17)):
            m=row['per_seed'][i];r=reference['per_seed'][i]
            scores=relative_scores(np.asarray(m['horizon_micro_rmse_mm'])**2,np.asarray(r['horizon_micro_rmse_mm'])**2,np.asarray(r['horizon_bias_mm']))
            per_seed.append(dict(seed=seed,**{k:v.tolist() for k,v in scores.items()}))
        rows.append(dict(name=row['name'],predictive_r2_by_horizon=np.mean([r['predictive_r2'] for r in per_seed],axis=0).tolist(),
            relative_residual_energy_reduction_by_horizon=np.mean([r['relative_residual_energy_reduction'] for r in per_seed],axis=0).tolist(),per_seed=per_seed))
    write_json(out/'residual_energy_analysis.json',plain(dict(rows=rows,summary_sha256=sha(out/'summary.json'),code_sha256=sha(__file__),
        role='post-fit2019 diagnostic scores only; no change to coefficients, normalization or source predictions',
        definitions=dict(predictive_r2='1-candidate_microMSE/variance_of_frozen_TF_residual_about_its2019_mean',
            relative_residual_energy_reduction='1-candidate_microMSE/frozen_TF_microMSE'),
        strict_oof=False,confirmation_C_D_read=False)))
    lines=['# 残差预测R²与能量减少率','',
        '2019探针迁移评价，基础p14/TF权重已拟合2019。每个种子先计算评分，再平均两种子；评分分母使用2019标签仅用于评价，不进入探针拟合或归一化。方差指TF残差围绕2019均值的方差，不是原始ZWD场的总方差。',
        '', '| 探针 | 当前预测R² | 六小时预测R² | 当前残差能量减少率 | 六小时残差能量减少率 |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['name']} | {r['predictive_r2_by_horizon'][0]:.6f} | {r['predictive_r2_by_horizon'][6]:.6f} | {r['relative_residual_energy_reduction_by_horizon'][0]:.6f} | {r['relative_residual_energy_reduction_by_horizon'][6]:.6f} |")
    lines+=['','预测R²＝1−修正后MSE/中心化目标残差方差；能量减少率＝1−修正后MSE/原TF残差MSE。原残差存在均值偏差时，两者不同，none的预测R²可以为负。零方差时R²记NA。小数比例可乘100%表示百分比。',
        '', '[主报告、配对区间与分层](REPORT.md)；[精确值与来源](residual_energy_analysis.json)。']
    (out/'RESIDUAL_ENERGY.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'));print(rows,flush=True)


if __name__=='__main__':main()
