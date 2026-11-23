"""Publish a measured feasibility verdict after confirmations, never before."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gnss_field.artifacts import read_json, write_json, sha256

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'results/20261123_generator_mechanism_validation_v01'


def main():
    confirmation = read_json(ROOT/'confirmation_results.json')
    if confirmation['status'] != 'COMPLETE':
        raise ValueError('Confirmation is incomplete')
    config = read_json(ROOT/'experiment.lock.json')
    reference_stability = read_json(ROOT/'reference_stability.json')
    if reference_stability['status'] != 'COMPLETE':
        raise ValueError('Previous-product stability checks are incomplete')
    original = read_json(PROJECT/'results/20261123_enhanced_encoder_generative_v03/generators.lock.json')
    for name, path in [('sampling_code', 'src/gnss_field/models/diffusion.py'),
                       ('data_loader', 'tools/simple_weather_unet_pilot.py'),
                       ('evaluation_code', 'src/gnss_field/evaluation/metrics.py')]:
        if sha256(PROJECT/path) != original[name]:
            raise ValueError(f'Original dependency changed: {name}')
    audit = read_json(ROOT/'preflight.json')['audits']
    data = np.load(PROJECT/'results/20261123_simple_weather_unet_v02/data.npz')
    exposure = {}
    for path in sorted(ROOT.glob('seed*/*/*/last.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        schedule = checkpoint['schedule'][:checkpoint['step']]
        count = int(data['mask'][schedule.ravel()][:, data['split_fit']].sum())
        exposure[str(path.relative_to(ROOT))] = dict(updates=checkpoint['step'], hour_instances=int(schedule.size),
                    unique_hours=int(np.unique(schedule).size), supervised_station_hours=count,
                    checkpoint_sha256=sha256(path), source_step=checkpoint['source_step'])
    verdicts = {}
    prefix_errors = {}
    for high_path in ROOT.glob('endpoints64/*/*.npz'):
        low_path = ROOT/'endpoints32'/high_path.parent.name/high_path.name
        high, low = np.load(high_path), np.load(low_path)
        errors = []
        for panel in ('training', 'withheld_weather'):
            np.testing.assert_array_equal(high[f'{panel}_hours'], low[f'{panel}_hours'])
            np.testing.assert_array_equal(high[f'{panel}_mask'], low[f'{panel}_mask'])
            errors.append(float(abs(high[f'{panel}_samples'][:, :32]-low[f'{panel}_samples']).max()))
        if max(errors) > .001:
            raise ValueError('Actual endpoint sample prefix changed by more than0.001mm')
        prefix_errors[str(high_path.relative_to(ROOT))] = max(errors)
    lines = ['# 三类模型方法验证结论', '',
             '本机RTX4060，真实GNSS拟合B站监督；每臂新增300次更新，包含原优化器状态的配对适配。',
             '这次评价使用固定训练24小时与内部留出24小时的四格诊断。C2019未用于本轮选型或评分，D2020封存。', '',
             '|改动|K32对照RMSE|K32干预RMSE|差值|对照宏CRPS|干预宏CRPS|方向筛查|',
             '|---|---:|---:|---:|---:|---:|---|']
    names = dict(vae='移除VAE后期ELBO', gan='关闭当前GAN对抗项', diffusion='Diffusion增加无偏均值MSE')
    for family in ('vae', 'gan', 'diffusion'):
        record = confirmation['families'][family]
        result = record['primary']
        control = result['records']['control']['scores']['withheld_weather_withheld']
        intervention = result['records']['intervention']['scores']['withheld_weather_withheld']
        effect_confirmed = record['status'] == 'SHORT_ADAPTATION_DIRECTION_CONFIRMED'
        practical_point = effect_confirmed and record.get('point_confirmed', False) and record['intervention_vs_previous_point_rmse'] < 0
        stable_reference = reference_stability['families'].get(family, {}).get('previous_joint_advantage_stable', False)
        practical_probability = effect_confirmed and record.get('probability_confirmed', False) and record['intervention_vs_previous_joint']['probability_screen'] and stable_reference
        if practical_point or practical_probability:
            verdict = '本机短适配方向可行；可进入更大数据验证，未证明完整链条优势'
        elif effect_confirmed:
            verdict = '后期微调策略的配对收益已复核；暂不替换原有最优产品' if family == 'gan' else '概率收益伴随点值损失；未超过原有优选，本轮不采用'
        elif record['status'] == 'INSUFFICIENT_REPLICATION_BUDGET':
            verdict = '复验未完成，证据不足'
        elif record['status'] == 'DIRECTION_NOT_STABLE':
            verdict = '复验不稳定，本轮不采用'
        else:
            verdict = '未达到预登记收益门槛，本轮不采用'
        verdicts[family] = dict(verdict=verdict, engineering_runnable=True, adaptation_effect_confirmed=effect_confirmed,
                               practical_point_gain=bool(practical_point), practical_probability_gain=bool(practical_probability))
        lines.append(f'|{names[family]}|{control["macro_rmse"]:.4f}|{intervention["macro_rmse"]:.4f}|'
                     f'{result["delta_macro_rmse_mm"]:+.4f}|{control["macro_CRPS_mm"]:.4f}|{intervention["macro_CRPS_mm"]:.4f}|{verdict}|')
    lines += ['', '所有差值为干预−配对对照；负误差差值表示改善。评价成员、噪声前缀与标签/支持掩码一致。', '', '## 逐模型证据', '']
    for family in ('vae', 'gan', 'diffusion'):
        r = confirmation['families'][family]
        p = r['primary']
        lines += [f'### {names[family]}', '', verdicts[family]['verdict']+'。',
                  f'主对照站组ES相对变化：{p["relative_group_ES_change"]*100:+.3f}%；宏CRPS相对变化：{p["relative_macro_CRPS_change"]*100:+.3f}%。',
                  f'RMSE配对天气块95%区间：{p["delta_macro_rmse_ci95"]}；宏CRPS区间：{p["delta_macro_crps_ci95"]}。']
        if 'adaptation_seed17' in r:
            for name in ('adaptation_seed17', 'independent_noise', 'high_k64'):
                v = r[name]
                lines.append(f'{name}：RMSE差{v["delta_macro_rmse_mm"]:+.4f}mm，站组ES变化{v["relative_group_ES_change"]*100:+.3f}%，宏CRPS变化{v["relative_macro_CRPS_change"]*100:+.3f}%。')
            lines.append(f'与原point_best同K32比较，RMSE差{r["intervention_vs_previous_point_rmse"]:+.4f}mm；与原joint_best的配对对比保留于confirmation_results.json。')
            old_joint = r['intervention_vs_previous_joint']
            lines.append(f'相对原joint_best：RMSE差{old_joint["delta_macro_rmse_mm"]:+.4f}mm，站组ES变化{old_joint["relative_group_ES_change"]*100:+.3f}%，宏CRPS变化{old_joint["relative_macro_CRPS_change"]*100:+.3f}%。')
            stability = reference_stability['families'][family]
            for key in ('k64', 'independent_noise'):
                v = stability['comparisons'][key]
                lines.append(f'旧joint_best的{key}匹配对照：RMSE差{v["delta_macro_rmse_mm"]:+.4f}mm，站组ES变化{v["relative_group_ES_change"]*100:+.3f}%，宏CRPS变化{v["relative_macro_CRPS_change"]*100:+.3f}%。')
            lines.append('相对既有优选的全部采样复核是否均通过预登记概率门槛：'+str(stability['previous_joint_advantage_stable'])+'。该门槛是本轮筛查规则，不是方法可行性的物理分界。')
        lines.append('')
    kl = audit['vae']['records']
    lines += ['## 机制诊断', '',
              f'VAE在8个固定拟合时刻的原始KL为{min(v["raw_KL"] for v in kl):.2f}—{max(v["raw_KL"] for v in kl):.2f}nats，'
              f'先验成员站值标准差为{min(v["prior_member_std_mm"] for v in kl):.2f}—{max(v["prior_member_std_mm"] for v in kl):.2f}mm，'
              '解码对潜变量有明显响应，不能解释为完全忽略潜变量。后验使用拟合B标签，重构优势不作为部署成绩。',
              f'GAN四个固定拟合批次的加权对抗梯度/评分梯度范数比为{audit["gan"]["weighted_adversarial_gradient_ratio"]}；'
              '梯度方向关系只作为定位依据，方法结论以配对预测为准。', '',
              '## 验证范围', '',
              '这轮真正验证的是三个单因素适配改动，不是重新从头训练三类网络。种子17是适配阶段复核，源模型相同。',
              '区间来自12个月份天气块同步重采样2000次，不是全年7日块bootstrap；独立噪声复查另列，二者不能互相替代。',
              '未通过表示本数据、起点、损失系数和短预算下不支持采用该改动，不否定整个GAN/VAE/Diffusion类别。',
              '显式双支路等其他结构尚未在这次试验中训练，因此不将其列为已验证可行的方法，也不据此启动服务器正式训练。',
              '工程验收、来源哈希、训练曝光与完整指标分别见preflight.json、experiment.lock.json、exposure.json、confirmation_results.json。', '']
    (ROOT/'VERDICT.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(ROOT/'exposure.json', exposure)
    write_json(ROOT/'actual_prefix_check.json', dict(tolerance_mm=.001, errors=prefix_errors))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for y, family in enumerate(('vae', 'gan', 'diffusion')):
        primary = confirmation['families'][family]['primary']
        for ax, value, interval in [(axes[0], primary['delta_macro_rmse_mm'], primary['delta_macro_rmse_ci95']),
                                    (axes[1], primary['records']['intervention']['scores']['withheld_weather_withheld']['macro_CRPS_mm']-
                                     primary['records']['control']['scores']['withheld_weather_withheld']['macro_CRPS_mm'], primary['delta_macro_crps_ci95'])]:
            ax.hlines(y, interval[0], interval[1], color='#527fac', linewidth=2)
            ax.plot(value, y, 'o', color='#245782')
        axes[2].plot(primary['relative_group_ES_change']*100, y, 'o', color='#245782')
    for index, (ax, title, label) in enumerate(zip(axes, ['Point error', 'Marginal probability', 'Joint probability'],
                               ['Delta macro RMSE (mm)', 'Delta macro CRPS (mm)', 'Delta group ES (%)'])):
        ax.axvline(0, color='grey', linestyle='--', linewidth=1)
        ax.set(title=title, xlabel=label, yticks=range(3), yticklabels=['VAE: remove ELBO', 'GAN: remove adversarial', 'DF: add mean U-MSE'] if index == 0 else ['', '', ''], ylim=(-.5, 2.5))
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('K32 paired differences; negative is better. RMSE/CRPS lines:95% monthly-block intervals.', fontsize=12)
    fig.savefig(ROOT/'paired_delta.png', dpi=180)
    plt.close(fig)
    write_json(ROOT/'validation_complete.json', dict(status='COMPLETE', verdicts=verdicts,
               cumulative_seconds=read_json(ROOT/'local_budget.json')['consumed_seconds'],
               verdict_sha256=sha256(ROOT/'VERDICT.md'), confirmation_sha256=sha256(ROOT/'confirmation_results.json'),
               server_jobs_submitted=False))
    write_json(ROOT/'status.json', dict(stage='VALIDATION_COMPLETE', verdicts=verdicts,
               replacement_supported=any(v['practical_point_gain'] or v['practical_probability_gain'] for v in verdicts.values()),
               server_jobs_submitted=False))
    print('MEASURED_VERDICT_PUBLISHED', json.dumps(verdicts, ensure_ascii=False))


if __name__ == '__main__':
    main()
