"""Audit and plot the completed, matched local U-Net experiment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simple_weather_unet_pilot import ROOT, DATA, PROJECT, metrics, PilotData, SimpleUNet
from gnss_field.artifacts import read_json, write_json, sha256
from gnss_field.data import PackedStore
from gnss_field.geometry.operators import surface_query, safe_coarsening_mask


def audit():
    a = np.load(ROOT/'data.npz', allow_pickle=False)
    lock = read_json(ROOT/'experiment.lock.json')
    receipt = read_json(ROOT/'data_complete.json')
    assert receipt['sha256'] == sha256(ROOT/'data.npz')
    assert lock['script_sha256'] == receipt['script_sha256']
    times = a['hours_train']
    for start in lock['calendar']['inner_starts']+lock['calendar']['outer_starts']:
        assert not ((times >= start-18) & (times < start+72+18)).any()
    store = PackedStore(DATA, purpose='train')
    sets = [set(store.stations.loc[a[f'split_{name}'], 'dense_group']) for name in ('fit', 'inner', 'outer')]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    try:
        store.targets('D', [int(np.datetime64('2020-01-01', 'h').astype('i8'))])
    except PermissionError:
        pass
    else:
        raise AssertionError('Retrospective read unexpectedly allowed')
    field = np.arange(144*96, dtype='f4').reshape(144, 96)/50
    via_matrix = (field.ravel()[a['qindices']]*a['qweights']).sum(-1)*np.exp(-a['heights']/3000)
    via_operator = surface_query(torch.from_numpy(field), a['qxy'], a['heights']).numpy()
    difference = float(np.max(abs(via_matrix-via_operator)))
    assert difference <= .001
    # Use the actual condition path, then perturb every loaded label and mask.
    data = PilotData()
    rows = data.rows('train')[:2]
    first, base = data.batch(rows, True)
    saved_labels, saved_mask = data.labels.clone(), data.mask.clone()
    data.labels.add_(10000)
    data.mask.logical_not_()
    second, base2 = data.batch(rows, True)
    assert torch.equal(first, second) and torch.equal(base, base2)
    data.labels.copy_(saved_labels)
    data.mask.copy_(saved_mask)
    report = dict(passed=True, data_sha256=receipt['sha256'],
                  query_max_abs_mm=difference, label_perturbation_input_change=0.,
                  full_weather_block_buffer_hours=18, spatial_groups_disjoint=True,
                  D2020_read_denied=True, parameters=124529,
                  input_source='A-only sampled meteorology; not independent full gridded weather',
                  stats='A training hours only; no inherited B-trained statistics or weights')
    write_json(ROOT/'acceptance.json', report)
    del data
    torch.cuda.empty_cache()
    print(json.dumps(report), flush=True)


def bootstrap_gain(pred, reference, y, mask, hours, replicates=2000):
    month = np.asarray(hours).astype('datetime64[h]').astype('datetime64[M]')
    blocks = np.unique(month)
    if len(blocks) < 2:
        return None
    chunks = [np.flatnonzero(month == block) for block in blocks]
    error = np.where(mask, pred-y, 0.).astype('f8')
    ref_error = np.where(mask, reference-y, 0.).astype('f8')
    counts = np.stack([mask[rows].sum(0) for rows in chunks])
    squared = np.stack([(error[rows]**2).sum(0) for rows in chunks])
    ref_squared = np.stack([(ref_error[rows]**2).sum(0) for rows in chunks])
    rng = np.random.default_rng(20261123)
    weights = rng.multinomial(len(blocks), np.full(len(blocks), 1/len(blocks)), size=replicates)
    n = weights@counts
    with np.errstate(invalid='ignore', divide='ignore'):
        candidate = np.sqrt(np.where(n > 0, weights@squared/n, np.nan))
        original = np.sqrt(np.where(n > 0, weights@ref_squared/n, np.nan))
    differences = np.nanmean(candidate, axis=1)-np.nanmean(original, axis=1)
    return dict(difference_mm=metrics(pred, y, mask)['macro_rmse']-metrics(reference, y, mask)['macro_rmse'],
                ci95_mm=np.percentile(differences, [2.5, 97.5]).tolist(), blocks=len(blocks),
                resampling='paired monthly event blocks, all stations synchronous', repetitions=replicates)


def pair_difference(pred, target, mask, xy, heights=None):
    i, j = np.triu_indices(len(xy), 1)
    distance = np.linalg.norm(xy[i]-xy[j], axis=1)/1000
    results = {}
    for lo, hi in ((0, 25), (25, 50), (50, 100), (100, 200)):
        selected = (distance > lo) & (distance <= hi)
        if heights is not None:
            selected &= abs(heights[i]-heights[j]) <= 100
        ii, jj = i[selected], j[selected]
        m = mask[:, ii] & mask[:, jj]
        error = (pred[:, ii]-pred[:, jj])-(target[:, ii]-target[:, jj])
        results[f'{lo}_{hi}km'] = dict(station_pairs=int(m.any(0).sum()), pair_hours=int(m.sum()),
                                      rmse_mm=float(np.sqrt(np.mean(error[m]**2))) if m.any() else None)
    return results


def report():
    runs = read_json(ROOT/'results.json')['runs']
    a = np.load(ROOT/'data.npz', allow_pickle=False)
    safe = safe_coarsening_mask(a['land']).numpy().ravel()[a['qindices']].all(1)
    plots = ROOT/'figures'
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for run in runs:
        curve = read_json(ROOT/run['name']/'curve.json')
        label = f'{"GNSS+weather" if run["weather"] else "GNSS only"}, seed {run["seed"]}'
        axes[0].plot([c['step'] for c in curve], [c['train']['macro_rmse'] for c in curve], label=label)
        axes[1].plot([c['step'] for c in curve], [c['inner']['macro_rmse'] for c in curve], label=label)
    for ax, title in zip(axes, ['Fixed fitting stations / hours', 'Held-out stations AND weather blocks']):
        ax.set(title=title, xlabel='Updates', ylabel='Station-macro RMSE (mm)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(plots/'learning_curves.png', dpi=180)
    plt.close(fig)
    confirmations, paired = {}, {}
    for run in runs:
        item = {}
        for cohort in ('spatial', 'temporal', 'spatial_temporal', 'C2019_development'):
            z = np.load(ROOT/run['name']/f'{cohort}_predictions.npz')
            pred, y, mask = z['prediction'], z['target'], z['mask']
            ids = z['station_ids']
            fm = mask & safe[ids][None]
            height = a['heights'][ids]
            to_z0 = np.exp(height/3000)
            item[cohort] = dict(vs_IDW=bootstrap_gain(pred, z['baseline'][:, 1], y, mask, z['hour_ids']),
                                pairs_Z0=pair_difference(pred*to_z0, y*to_z0, mask, a['xy'][ids]),
                                pairs_surface_height_difference_le100m=pair_difference(pred, y, mask, a['xy'][ids], height))
            if fm.any():
                item[cohort]['fine_scale_common_mask'] = dict(
                    fine=metrics(pred, y, fm), coarse=metrics(z['coarsened'], y, fm),
                    pairs_Z0_fine=pair_difference(pred*to_z0, y*to_z0, fm, a['xy'][ids]),
                    pairs_Z0_coarse=pair_difference(z['coarsened']*to_z0, y*to_z0, fm, a['xy'][ids]),
                    pairs_surface_fine=pair_difference(pred, y, fm, a['xy'][ids], height),
                    pairs_surface_coarse=pair_difference(z['coarsened'], y, fm, a['xy'][ids], height))
        confirmations[run['name']] = item
    for seed in sorted(set(r['seed'] for r in runs)):
        off = next(r for r in runs if r['seed'] == seed and not r['weather'])
        on = next(r for r in runs if r['seed'] == seed and r['weather'])
        paired[str(seed)] = {}
        for cohort in ('spatial', 'temporal', 'spatial_temporal', 'C2019_development'):
            p = np.load(ROOT/on['name']/f'{cohort}_predictions.npz')
            q = np.load(ROOT/off['name']/f'{cohort}_predictions.npz')
            assert np.array_equal(p['mask'], q['mask']) and np.array_equal(p['hour_ids'], q['hour_ids'])
            paired[str(seed)][cohort] = bootstrap_gain(p['prediction'], q['prediction'], p['target'], p['mask'], p['hour_ids'])
    write_json(ROOT/'paired_diagnostics.json', dict(runs=confirmations, weather_minus_no_weather=paired,
               interval_scope='monthly event blocks within each population, not a full-year 7-day bootstrap'))
    labels = ['A coarse', 'A IDW', 'ERA5 sampled', 'U-Net no weather', 'U-Net + weather']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, cohort in zip(axes, ('spatial', 'temporal', 'spatial_temporal')):
        values = []
        for name in ('A_coarse_bilinear', 'A_IDW_grid', 'ERA5_station_PWV_Tm_grid'):
            values.append(np.mean([r['results'][cohort][name]['macro_rmse'] for r in runs]))
        for weather in (False, True):
            values.append(np.mean([r['results'][cohort]['unet']['macro_rmse'] for r in runs if r['weather'] == weather]))
        ax.bar(np.arange(5), values, color=['#b5c2cc', '#93a8bb', '#b9b0a4', '#6895bf', '#218f80'])
        ax.set(xticks=np.arange(5), xticklabels=labels, title=cohort.replace('_', ' '), ylabel='Station-macro RMSE (mm)')
        ax.tick_params(axis='x', labelrotation=55)
        ax.grid(axis='y', alpha=.2)
        for i, v in enumerate(values):
            ax.text(i, v+.06, f'{v:.3f}', ha='center', fontsize=8)
    fig.savefig(plots/'holdout_comparison.png', dpi=180)
    plt.close(fig)
    # Representative event fixed by calendar: first outer hour, seed42 model.
    data = PilotData()
    row = data.rows('outer')[:1]
    fields = []
    for weather in (False, True):
        name = f'{"gnss_weather_terrain" if weather else "gnss_terrain"}_seed42'
        model = SimpleUNet().cuda()
        model.load_state_dict(torch.load(ROOT/name/'best.pt', map_location='cuda', weights_only=False)['model'])
        model.eval()
        with torch.no_grad():
            x, base = data.batch(row, weather)
            field = (base+10*model(x))[0, 0].cpu().numpy()
        if not fields:
            fields.append(base[0, 0].cpu().numpy())
        fields.append(field)
    land = np.asarray(a['land'], bool)
    np.save(ROOT/'example_fields_Z0.npy', np.stack(fields))
    reloaded = np.load(ROOT/'example_fields_Z0.npy')
    reload_differences = {}
    for k, weather in enumerate((False, True), start=1):
        name = f'{"gnss_weather_terrain" if weather else "gnss_terrain"}_seed42'
        archived = np.load(ROOT/name/'spatial_temporal_predictions.npz')
        assert archived['hour_ids'][0] == data.hours[row[0]]
        ids = archived['station_ids']
        queried = surface_query(torch.from_numpy(reloaded[k]), a['qxy'][ids], a['heights'][ids]).numpy()
        discrepancy = float(np.max(abs(queried-archived['prediction'][0])))
        assert discrepancy <= .001
        reload_differences[name] = discrepancy
    write_json(ROOT/'published_reload_check.json', dict(passed=True, max_abs_mm=reload_differences,
               product='example_fields_Z0.npy', units='common_height_Z0_mm'))
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)
    vmin, vmax = np.percentile(fields[0][land], [2, 98])
    for ax, field, title in zip(axes, fields, ['Coarse A field', 'U-Net without weather', 'U-Net with weather']):
        m = ax.imshow(np.where(land, field, np.nan), origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set(title=title, xlabel='25 km grid column', ylabel='25 km grid row')
    fig.colorbar(m, ax=axes, label='Common-height Z0 (mm)')
    fig.suptitle(f'Fixed event: {data.hours[row[0]].astype("datetime64[h]")}; illustrative field, not gridded truth')
    fig.savefig(plots/'example_fields.png', dpi=180)
    plt.close(fig)
    lines = ['# 本机简单U-Net结果', '', '用户确认使用现有数据。气象为A站产品中的ERA5采样，非新增完整气象格网。', '',
             '模型124,529参数，宽度16/32/64，从头训练；两种输入、两个种子，各1000步。B内部选模，C仅作开发确认，D未读取。', '',
             '|模型|种子|选中步|留站RMSE|留天气RMSE|留站且留天气RMSE|C2019小子集RMSE|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        vals = [r['results'][c]['unet']['macro_rmse'] for c in ('spatial', 'temporal', 'spatial_temporal', 'C2019_development')]
        lines.append(f'|{"GNSS+气象+地形" if r["weather"] else "GNSS+地形"}|{r["seed"]}|{r["selected_step"]}|'+ '|'.join(f'{v:.4f}' for v in vals)+'|')
    lines += ['', '## 简单对照（相同评分人口）', '', '|方法|留站|留天气|留站且留天气|C2019小子集|', '|---|---:|---:|---:|---:|']
    for name in ('A_coarse_bilinear', 'A_IDW_grid', 'ERA5_station_PWV_Tm_grid'):
        vals = [runs[0]['results'][c][name]['macro_rmse'] for c in ('spatial', 'temporal', 'spatial_temporal', 'C2019_development')]
        lines.append('|'+name+'|'+'|'.join(f'{v:.4f}' for v in vals)+'|')
    lines += ['', '## 解释范围', '',
              '- 留站：48个拟合天气时刻上的B外层77站；留天气：2018每月16—18日、每6小时，共144时刻的拟合站；二者交叉单列。实际可评分站及站时刻数见results.json。',
              '- C2019为每月4个固定时刻，共48小时；不是全年，不能与旧全年数字直接排序。',
              '- 月份事件块配对区间用于小试诊断，不代替完整全年7日块bootstrap。两种子是优化稳定性检查，不是独立气象样本。',
              '- 100km对照的细尺度判断使用完整陆地粗化模板的共同掩码，详见paired_diagnostics.json；不使用海陆模板不完整的站点支持细尺度结论。',
              '- 当前模型为确定性U-Net，不产生GAN/扩散/VAE概率结果。是否扩展编码器取决于跨站和跨天气的配对收益，而非训练误差单独下降。',
              '', '图件：figures/learning_curves.png、holdout_comparison.png、example_fields.png。', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('LOCAL_UNET_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit()
    if not args.audit_only:
        report()
