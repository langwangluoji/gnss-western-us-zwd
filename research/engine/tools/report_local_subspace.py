"""Render the completed pilot evidence without fitting or selecting any model."""
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'results/20261119_local_edm_subspace_v01'


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history(folder, filename):
    return [json.loads(line) for line in (ROOT / folder / filename).read_text().splitlines() if line]


def main():
    annual = read('full2019/results.json')
    attr = read('full2019/attribution_results.json')
    control = read('mean_shift_control_v02/annual_results.json')
    audit = read('review/annual_server_audit.json')
    release = read('server_release.json')
    figures = ROOT / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'EDM': '#3274a1', 'Mean-only Gaussian': '#d47725'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for label, folder in [('EDM', 'scientific'), ('Mean-only Gaussian', 'mean_shift_control_v02')]:
        dev = history(folder, 'development.jsonl')
        initial = read(folder + '/initial_metrics.json')['joint']
        steps = [0] + [row['step'] for row in dev]
        axes[0].plot(steps, [initial['value']] + [r['metrics']['joint']['value'] for r in dev],
                     'o-', label=label, color=colors[label], markersize=3)
        axes[1].plot(steps, [initial['ensemble_mean_rmse_mm']] +
                     [r['metrics']['joint']['ensemble_mean_rmse_mm'] for r in dev],
                     'o-', label=label, color=colors[label], markersize=3)
        train = history(folder, 'training.jsonl')
        axes[2].plot([r['step'] for r in train], [r['loss'] for r in train],
                     'o-', label=label, color=colors[label], markersize=3)
    for ax, title, ylabel in zip(axes,
            ['B2018 development ES (K=16)', 'B2018 finite-ensemble mean RMSE', 'Training ES, different hourly batches'],
            ['ES (mm), lower is better', 'Station macro RMSE (mm)', 'Training ES (mm)']):
        ax.set(title=title, xlabel='Optimizer updates', ylabel=ylabel)
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False)
    fig.savefig(figures / 'training_curves.png', dpi=180)
    plt.close(fig)

    names = ['Gaussian', 'Mean-only\nGaussian', 'EDM', 'Gaussian +\nEDM mean shift', 'EDM minus\nmean shift']
    entries = [attr['joint']['Gaussian'], control['joint']['mean_control'], attr['joint']['EDM'],
               attr['joint']['mean_only_Gaussian'], attr['joint']['shape_only_EDM']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, key in zip(axes, ['ES_mm', 'VS_mm']):
        vals = [e[key] for e in entries]
        ax.plot(range(5), vals, 'o', color='#3274a1', markersize=8)
        ax.axhline(vals[0], color='#777777', linestyle=':', linewidth=1)
        for i, value in enumerate(vals):
            ax.annotate(f'{value:.6f}', (i, value), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
        span = max(vals) - min(vals)
        ax.set_ylim(min(vals) - .4 * span, max(vals) + .5 * span)
        ax.set_xticks(range(5), names)
        ax.set(ylabel=key.replace('_mm', ' (mm)'), title='C2019, 6 dense station groups; lower is better')
        ax.grid(axis='y', alpha=.18)
    fig.savefig(figures / 'annual_attribution.png', dpi=180)
    plt.close(fig)

    contrasts = audit['annual']['crossed']
    keys = ['EDM_minus_Gaussian', 'EDM_minus_Gaussian_plus_independent_EDM_shift',
            'EDM_minus_independent_EDM_shift_minus_Gaussian', 'trained_mean_control_minus_Gaussian',
            'trained_mean_control_minus_EDM']
    labels = ['EDM - Gaussian', 'EDM - Gaussian with EDM mean', 'Shape-only EDM - Gaussian',
              'Mean control - Gaussian', 'Mean control - EDM']
    vals = np.array([contrasts[k]['difference_mm']['ES'] for k in keys])
    ci = np.array([contrasts[k]['7']['intervals_mm']['ES'] for k in keys])
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    ax.errorbar(vals, range(5), xerr=np.array([vals - ci[:, 0], ci[:, 1] - vals]),
                fmt='o', capsize=4, color='#3274a1')
    ax.axvline(0, color='#777777', linestyle=':')
    ax.set_yticks(range(5), labels)
    ax.invert_yaxis()
    ax.set(xlabel='Paired ES difference (mm), negative favors first model',
           title='C2019: 2000 paired 7-day block x station-group bootstrap draws')
    ax.grid(axis='x', alpha=.18)
    fig.savefig(figures / 'annual_ES_intervals.png', dpi=180)
    plt.close(fig)

    report = f'''# R64残差生成小试：全年开发结果与服务器安排

本机实测支持把下一轮重点放在条件均值校正与天气覆盖。EDM的全年ES改善主要由均值移动解释，当前证据尚未建立额外的非高斯空间结构收益。保留EDM作为同条件对照，比立即增加训练时长更有助于确定研究方向。

## 数据与比较口径

输入为A174站，新增头使用2018年B511站监督。本机小试使用576个训练小时和288个整块留出小时；冻结底座此前已使用全部864个B小时，因此称为冻结底座上的增量开发验证。条件PCA只在576个头训练小时拟合。

C2019已用于研究开发，本次全年点值评价覆盖8760小时、76站、642,258站时。联合评分覆盖6个密集站组。D2020保持封存。25 km表示发布格网间距，细尺度真实性仍需独立站差值与100 km回采样对照支持。

M0固定为第5000步EMA。发布Z0均值先与A站IDW按固定训练期权重混合，再叠加残差。所有方法经过同一格网查询与站高恢复。条件高斯采用冻结局部空间基和A条件振幅；EDM只更新其R64可观测敏感系数，其余空间成分及站端未解析误差保持高斯。

## 全年2019点值结果

| 产品 | 站宏RMSE，mm | 均值计算方式 |
|---|---:|---|
| 冻结M0 | 10.344629 | 确定性格网查询 |
| A站IDW格网 | 10.278529 | 确定性格网查询 |
| 固定M0/IDW混合均值 | {annual['analytic_point']['station_macro_rmse_mm']:.6f} | 解析均值 |
| 独立训练的均值头＋条件高斯 | {control['exact_point']['station_macro_rmse_mm']:.6f} | 解析均值 |
| EDM均值 | {attr['independent_integrated_mean_point']['station_macro_rmse_mm']:.6f} | 独立128成员配对积分估计 |

EDM积分池与评分成员使用不同随机种子，估计误差标准误RMS为{attr['shift_estimation_SE_RMS_mm']:.6f} mm。64与128前缀的点值RMSE差约0.000003 mm。原始有限128成员均值的EDM RMSE为10.004367 mm，配对高斯为10.032252 mm；该采样误差口径与上表的解析/积分均值分列。

## 联合概率结果与收益归因

| 产品 | ES，mm | VS，mm |
|---|---:|---:|
| 条件高斯 | 6.923941 | 1.367433 |
| 独立训练的均值头＋条件高斯 | 6.892246 | 1.366629 |
| EDM | 6.885321 | 1.367505 |
| 高斯＋独立池估计的EDM均值位移 | 6.885274 | 1.367506 |
| EDM去除该均值位移 | 6.924001 | 1.367431 |

ES为能量评分，衡量站组向量的联合预测；VS为变差评分，反映站间依赖，两者数值越低越好。使用所有站同步的7日块并对6个密集组交叉重采样，共2000次；14日敏感性结果方向一致。下表差值均为前者减后者，区间以时间与站组重采样口径为主。

| ES配对比较 | 差值，mm | 95%区间，mm |
|---|---:|---|
'''
    for label, key in zip(labels, keys):
        value = contrasts[key]
        lo, hi = value['7']['intervals_mm']['ES']
        report += f"| {label} | {value['difference_mm']['ES']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |\n"
    report += '''
EDM对齐均值后，形状贡献的区间跨零；均值头与EDM的差异区间也跨零，这不等于已证明两种模型等效。全年有限集合CRPS为高斯5.669051、均值头5.650800、EDM5.646141 mm。该经验V统计量含有限成员偏差，不能直接与此前解析高斯CRPS 5.628545 mm做优劣比较。

EDM的80/90/95%经验区间覆盖率约为77.67/86.15/90.71%，低于名义覆盖率。均值改善与区间校准是两个问题，下一轮继续同时记录覆盖率和区间宽度。

## 已实施网络与工程验证

- R64 EDM采用两层128维MLP、32维冻结条件、18步Heun约35次网络调用，真实站端K4无偏ES训练800步，EMA固定0.999。解析高斯初始化的有限步增益1.0447288398固定校正，不按开发误差重新拟合。
- 均值头采用匹配条件、隐藏宽度、初始化、小时顺序、优化器、EMA与800步预算，固定协方差，单独学习条件位置移动。两臂候选均由B开发ES选定第800步，没有用C重新选模。
- RTX4060本机EDM科学训练约61.7秒，均值头约7.4秒；基于已缓存背景和站端矩阵，耗时不含M0训练和全域格网生成。两臂实际速度差约8倍。
- Gaussian初始化、35调用反传、格网/站端等价、拆批及噪声前缀检查通过。实际连续200与100＋恢复100的模型、EMA、优化器、RNG及游标一致。
- 原始小试初值/拆批最大误差0.0000153 mm；服务器同入口本机基准初值误差0.00000763 mm，峰值显存约114 MB。RTX结果用于工程检查，不能替代真实A800预算验证。

## 与Pro讨论后的下一步

Pro的两轮实测讨论记录见[首轮](pro_response.md)与[全年归因后复核](pro_followup_response.md)。下一轮优先测试更广天气覆盖能否稳定改善条件位置。保留现有M0、空间先验、混合权重、条件振幅、PCA及R64投影，便于把变化归因于训练小时覆盖。

服务器包包含2018年B站全年条件和监督；12个完整开发周及两侧18小时缓冲排除后，有6288训练小时和固定288开发锚点。两遍数据、每次8小时，单臂最多1572次更新。冻结底座已见开发B天气的边界继续保留。默认顺序为真实A800短基准、预算锁、均值头、全年C2019评价。EDM可作为同数据同预算的一次覆盖反证，需登记原因。

单卡申请A800、2CPU、16GB；短基准上限5分钟，训练与评价各30分钟。这是墙钟上限，实际卡时由基准和作业记录确定。更长训练、GAN、Optuna及多种子不在本交接队列中。当前尚未上传或提交新的收费作业。

决策依据为全年点值、短距离差值、100 km回采样对照及ES/VS/CRPS与覆盖率。扩大覆盖后若收益仍可由均值解释，继续研究均值与校准；只有非高斯形状或空间依赖收益得到支持时，再扩大生成训练预算。没有将步数增加或降至6 mm写成可保证结果。

## 曲线、代码与交接

- [训练与B开发曲线](figures/training_curves.png)：不同训练批次的loss自然波动，开发指标使用固定小时；800步处仍下降只说明该短预算未出现平台，不能据此保证延长有益。
- [全年概率评分与均值归因](figures/annual_attribution.png)
- [ES时间块与站组配对区间](figures/annual_ES_intervals.png)
- [独立复核与数据清单](review/ANNUAL_AND_SERVER_REVIEW.md)
- [服务器命令与暂停恢复说明](../../docs/observable_server_handoff.md)
- [服务器包摘要及SHA256](server_release.json)
- [最终工程核验](verification.json)

封装包只包含本轮所需源码、冻结算子、B训练分片、独立C评价分片与Slurm脚本。原始失败记录、本机缓存及临时封装目录留在本机，不混入服务器发布包。
'''
    (ROOT / 'RESULTS.md').write_text(report, encoding='utf-8')

    archive = Path(release['archive'])
    assert digest(archive) == release['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        prefix = '20261119_observable_subspace_v01/'
        manifest = json.loads(z.read(prefix + 'release.json'))
        for relative, expected in manifest['files'].items():
            assert hashlib.sha256(z.read(prefix + relative)).hexdigest() == expected, relative
        assert len(z.namelist()) == len(manifest['files']) + 1
        assert not any('packaging_runtime' in name or '__pycache__' in name for name in z.namelist())
    m0 = PROJECT / 'results/20261118_us_dense_zwd_a800_m0_5k_evalfix_v02/seed_42/mean_base/candidate_005000.pt'
    assert digest(m0) == '605650e000ce4a0ab31d3d85996fd16316c978547fe9f2821c65c346cd1b86f0'
    for filename in ['selection.lock.json', 'mean_shift_control_v02/selection.lock.json']:
        lock = read(filename)
        assert digest(Path(lock['checkpoint'])) == lock['sha256']
    tests = ET.parse(ROOT / 'tests.xml').getroot()
    suites = list(tests.iter('testsuite'))
    test_count = sum(int(s.attrib['tests']) for s in suites)
    assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ['failures', 'errors'])
    verification = dict(archive_sha256=release['sha256'], archive_crc_passed=True,
        release_files_verified=len(manifest['files']), tests_passed=test_count,
        M0_sha256=digest(m0), selected_checkpoints_verified=True,
        training_role='B2018', evaluation_role='C2019', D2020_included=False,
        server_jobs_submitted=False, local_portable_benchmark=read('portable_final_edm/benchmark.json'),
        remaining_server_checks=['real Linux/A800 benchmark', 'Slurm signals and actual accounting', 'budget lock'],
        figures=[str(path.relative_to(ROOT)) for path in figures.glob('*.png')])
    (ROOT / 'verification.json').write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Report and 3 figures saved; {test_count} tests recorded; {len(manifest["files"])} release file hashes verified.')


if __name__ == '__main__':
    main()
