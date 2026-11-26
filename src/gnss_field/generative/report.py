"""CPU figures from final C-station archives, with an auditable CSV per diagnostic.

Only selected complete-model products and the deterministic control are opened.
Each checkpoint archive is reduced independently; ensemble quantiles use bounded
time chunks. No model loading, CUDA calls, or dense-truth inference occurs here.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np


LABELS = {'control': 'Deterministic control', 'complete_vae': 'Conditional VAE',
          'complete_gan': 'Conditional GAN', 'complete_diffusion': 'Conditional diffusion',
          'vae_representation': 'VAE representation', 'gan_pretrain': 'GAN latent pretraining',
          'diffusion_pretrain': 'Diffusion latent pretraining',
          'shared_vae': 'VAE shared representation', 'shared_gan': 'GAN shared representation',
          'shared_diffusion': 'Diffusion shared representation'}
FINAL_STAGES = ('control', 'complete_vae', 'complete_gan', 'complete_diffusion')
DISTANCE_EDGES_KM = (0., 25., 50., 100., 200.)


def _csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: ('' if isinstance(row.get(key), float) and
                                      not math.isfinite(row[key]) else row.get(key, ''))
                         for key in fields} for row in rows)
    return str(path.resolve())


def _station_values(truth, mean, background, mask):
    counts = mask.sum(0)
    def average(values):
        return np.divide(np.where(mask, values, 0.).sum(0), counts,
                         out=np.full(len(counts), np.nan), where=counts > 0)
    error = mean-truth
    baseline = np.sqrt(average((background-truth)**2))
    rmse = np.sqrt(average(error**2))
    return dict(valid_hours=counts, rmse_mm=rmse, mae_mm=average(abs(error)),
                bias_mm=average(error), background_rmse_mm=baseline,
                rmse_gain_mm=baseline-rmse)


def _intervals(samples, truth, mask):
    counts = mask.sum(0)
    use = counts > 0
    totals = {level: [np.zeros(mask.shape[1]), np.zeros(mask.shape[1])]
              for level in (80, 90, 95)}
    for start in range(0, len(truth), 168):
        stop = start+168
        chunk = samples[start:stop]
        if not np.isfinite(chunk).all():
            raise ValueError('Nonfinite archived probability prediction')
        for level, (hits, widths) in totals.items():
            alpha = (100-level)/200
            lo, hi = np.quantile(chunk, [alpha, 1-alpha], axis=1)
            m = mask[start:stop]
            hits += np.where(m, (truth[start:stop] >= lo) & (truth[start:stop] <= hi), 0).sum(0)
            widths += np.where(m, hi-lo, 0.).sum(0)
    return [dict(level=level, nominal_coverage=level/100,
                 coverage=float(np.mean(hits[use]/counts[use])),
                 width_mm=float(np.mean(widths[use]/counts[use])),
                 scored_stations=int(use.sum()), valid_station_hours=int(counts.sum()))
            for level, (hits, widths) in totals.items()]


def _differences(truth, mean, background, mask, xy, heights):
    first, second = np.triu_indices(len(xy), 1)
    distance = np.linalg.norm(xy[first]-xy[second], axis=1)/1000
    filters = [('all', np.ones(len(first), dtype=bool))]
    if heights is not None:
        filters.append(('difference_le_100m', abs(heights[first]-heights[second]) <= 100.))
    rows = []
    for name, height_mask in filters:
        for lo, hi in zip(DISTANCE_EDGES_KM[:-1], DISTANCE_EDGES_KM[1:]):
            selected = height_mask & (distance >= lo) & (distance < hi)
            # Include the registered upper endpoint in the last bin only.
            if hi == DISTANCE_EDGES_KM[-1]:
                selected |= height_mask & (distance == hi)
            ids = np.flatnonzero(selected)
            n = 0; squared = 0.; absolute = 0.; signed = 0.; baseline = 0.; pairs = 0
            # Pair chunks avoid a full hours x all-pairs allocation.
            for start in range(0, len(ids), 128):
                ii, jj = first[ids[start:start+128]], second[ids[start:start+128]]
                valid = mask[:, ii] & mask[:, jj]
                pairs += int(valid.any(0).sum()); n += int(valid.sum())
                error = (mean[:, ii]-mean[:, jj])-(truth[:, ii]-truth[:, jj])
                base = (background[:, ii]-background[:, jj])-(truth[:, ii]-truth[:, jj])
                squared += float(np.where(valid, error**2, 0.).sum())
                absolute += float(np.where(valid, abs(error), 0.).sum())
                signed += float(np.where(valid, error, 0.).sum())
                baseline += float(np.where(valid, base**2, 0.).sum())
            rmse = math.sqrt(squared/n) if n else float('nan')
            bg = math.sqrt(baseline/n) if n else float('nan')
            rows.append(dict(distance_min_km=lo, distance_max_km=hi, height_filter=name,
                geometric_pair_count=len(ids), pair_count=pairs, valid_pair_hours=n,
                difference_rmse_mm=rmse, difference_mae_mm=absolute/n if n else float('nan'),
                difference_bias_mm=signed/n if n else float('nan'), background_difference_rmse_mm=bg,
                difference_rmse_gain_mm=bg-rmse))
    return rows


def _paired_bootstrap(truth, mean, background, mask, xy, hours):
    """Paired circular time blocks, optionally crossed with 50-km group draws.

    Precomputed circular block SSE/count sums avoid revisiting individual
    station-hours within each of the 2,000 resamples. The same time draws are
    used for both predictors, both RMSE metrics, and every model archive.
    """
    if hours.shape != (len(truth),) or (len(hours) > 1 and not np.all(np.diff(hours) == 1)):
        raise ValueError('Paired bootstrap requires the complete ordered hourly calendar')
    selected = mask.any(0)
    truth, mean, background = truth[:, selected], mean[:, selected], background[:, selected]
    mask, xy = mask[:, selected], xy[selected]
    stats = np.stack((np.where(mask, (mean-truth)**2, 0.),
                      np.where(mask, (background-truth)**2, 0.), mask.astype('f8')), axis=-1)
    _, group = np.unique(np.floor(xy/50000.).astype('i8'), axis=0, return_inverse=True)
    group_count = int(group.max())+1
    ntime, nstation = mask.shape

    def gains(totals, weights):
        counts = totals[..., 2]
        weights = weights*(counts > 0)
        denominator = weights.sum(1)
        station_model = np.sqrt(np.divide(totals[..., 0], counts, out=np.zeros_like(counts), where=counts > 0))
        station_background = np.sqrt(np.divide(totals[..., 1], counts, out=np.zeros_like(counts), where=counts > 0))
        macro = np.divide(((station_background-station_model)*weights).sum(1), denominator,
                          out=np.full(len(totals), np.nan), where=denominator > 0)
        count = (counts*weights).sum(1)
        model = np.divide((totals[..., 0]*weights).sum(1), count, out=np.full(len(totals), np.nan), where=count > 0)
        base = np.divide((totals[..., 1]*weights).sum(1), count, out=np.full(len(totals), np.nan), where=count > 0)
        return np.stack((macro, np.sqrt(base)-np.sqrt(model)), axis=1)

    observed = gains(stats.sum(0)[None], np.ones((1, nstation)))[0]
    rows = []
    for days in (7, 14):
        block = min(24*days, ntime)
        full, remainder = divmod(ntime, block)
        def rolling(length):
            extended = np.concatenate((stats, stats[:length-1]), axis=0)
            cumulative = np.concatenate((np.zeros((1, nstation, 3)), extended.cumsum(0)), axis=0)
            return cumulative[length:length+ntime]-cumulative[:ntime]
        full_sums = rolling(block)
        tail_sums = rolling(remainder) if remainder else None
        draws = {'synchronous_time_blocks': [], 'time_blocks_and_50km_groups': []}
        rng = np.random.default_rng(20261122+days)
        group_rng = np.random.default_rng(20261122+1000+days)
        for start in range(0, 2000, 64):
            size = min(64, 2000-start)
            starts = rng.integers(0, ntime, size=(size, full+bool(remainder)))
            totals = full_sums[starts[:, :full]].sum(1)
            if remainder:
                totals += tail_sums[starts[:, -1]]
            draws['synchronous_time_blocks'].append(gains(totals, np.ones((size, nstation))))
            groups = group_rng.integers(0, group_count, size=(size, group_count))
            multiplicity = (groups[:, :, None] == np.arange(group_count)[None, None]).sum(1)
            draws['time_blocks_and_50km_groups'].append(gains(totals, multiplicity[:, group]))
        for scheme, batches in draws.items():
            distribution = np.concatenate(batches)
            for metric, column in (('station_macro_rmse', 0), ('micro_rmse', 1)):
                valid = distribution[:, column][np.isfinite(distribution[:, column])]
                lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
                rows.append(dict(block_days=days, actual_block_hours=block, resampling=scheme, metric=metric,
                    rmse_gain_mm=float(observed[column]), gain_ci95_low_mm=float(lo) if lo is not None else None,
                    gain_ci95_high_mm=float(hi) if hi is not None else None,
                    requested_replicates=2000, valid_replicates=len(valid), valid_station_hours=int(mask.sum()),
                    scored_stations=nstation, spatial_groups_50km=group_count, seed=20261122))
    return rows


def _reduce_archive(path, bootstrap=False):
    with np.load(path, allow_pickle=False) as archive:
        required = {'hours', 'truth', 'mean', 'background_mean', 'population', 'xy', 'samples'}
        missing = required-set(archive.files)
        if missing:
            raise ValueError('Missing archived arrays: '+', '.join(sorted(missing)))
        truth = np.asarray(archive['truth'], dtype='f8')
        mean = np.asarray(archive['mean'], dtype='f8')
        background = np.asarray(archive['background_mean'], dtype='f8')
        mask = np.asarray(archive['population'], dtype=bool)
        xy = np.asarray(archive['xy'], dtype='f8')
        heights = np.asarray(archive['heights'], dtype='f8') if 'heights' in archive.files else None
        if truth.ndim != 2 or mask.shape != truth.shape or mean.shape != truth.shape or background.shape != truth.shape:
            raise ValueError('Archived station-array shapes disagree')
        if xy.shape != (truth.shape[1], 2) or not np.isfinite(xy).all():
            raise ValueError('Invalid archived station coordinates')
        if heights is not None and (heights.shape != (truth.shape[1],) or not np.isfinite(heights).all()):
            raise ValueError('Invalid archived station heights')
        if not mask.any() or not np.isfinite(truth[mask]).all() or not np.isfinite(mean).all() or not np.isfinite(background).all():
            raise ValueError('Nonfinite archived forecast or no scored C observations')
        stations = _station_values(truth, mean, background, mask)
        differences = _differences(truth, mean, background, mask, xy, heights)
        bootstrap_started = time.monotonic()
        intervals = (_paired_bootstrap(truth, mean, background, mask, xy, np.asarray(archive['hours'], dtype='i8'))
                     if bootstrap else [])
        bootstrap_seconds = time.monotonic()-bootstrap_started
        # Keep only this archive's ensemble in memory, retaining its native dtype.
        samples = archive['samples']
        if samples.ndim != 3 or samples.shape[0] != len(truth) or samples.shape[2] != truth.shape[1] or samples.shape[1] < 2:
            raise ValueError('Invalid archived probability ensemble shape')
        calibration = _intervals(samples, truth, mask)
        members = int(samples.shape[1])
        del samples
    return dict(stations=stations, xy=xy, differences=differences, calibration=calibration,
                members=members, has_heights=heights is not None, bootstrap=intervals, bootstrap_seconds=bootstrap_seconds)


def _save(figure, path, figures, key):
    figure.savefig(path, dpi=160, bbox_inches='tight')
    figures[key] = str(path.resolve())
    # Figures are created through the OO Agg interface and have no GUI manager.
    figure.clear()


def export_campaign_figures(campaign) -> dict:
    """Export PNG/CSV artifacts for actual selected final C2019 products.

    ``campaign`` needs only ``root`` and ``manifest``. The returned JSON-safe
    manifest contains actual file paths and explicit reasons for skipped panels.
    Station differences pool valid pair-hours; interval scores average within
    each station and then across scored stations, as in score_ensemble.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    export_started = time.monotonic()
    final = campaign.manifest.get('stages', {}).get('final_evaluation', {})
    if final.get('status') != 'complete':
        raise ValueError('Final evaluation must be complete before figure export')
    if final.get('role', 'C') != 'C':
        raise ValueError('Figure export requires independent C-station evaluation')
    output = Path(campaign.root)/'evaluation'/'figures'
    output.mkdir(parents=True, exist_ok=True)
    result = dict(schema=1, evaluation_role='C', evaluation_year=final.get('year', 2019),
        figures={}, tables={}, archives=[], skipped=[],
        scope='Independent C-station diagnostics; output-grid spacing is not dense-field accuracy.',
        station_order='Archive station index; coordinates are projected metres.',
        interval_aggregation='Time mean within each station, then equal mean across scored stations.',
        difference_aggregation='RMSE/MAE/bias pooled over valid unordered station-pair hours; bins [lo,hi), final upper endpoint included.')

    def figure(*args, **kwargs):
        fig = Figure(*args, **kwargs)
        FigureCanvasAgg(fig)
        return fig

    # Losses are stage-specific objectives and therefore use separate panels.
    stages = campaign.manifest.get('stages', {})
    training_rows = []; development_rows = []; plotted = []
    for stage, entry in stages.items():
        history = entry.get('history', [])
        evaluation = entry.get('evaluation_history', [])
        if not history and not evaluation:
            continue
        plotted.append((stage, history, evaluation))
        for row in history:
            for metric, value in row.items():
                if metric not in ('step', 'epoch') and isinstance(value, (int, float)) and math.isfinite(value):
                    training_rows.append(dict(stage=stage, model=LABELS.get(stage, stage.replace('_', ' ')),
                        step=row['step'], epoch=row.get('epoch', ''), metric=metric, value=value))
        for row in evaluation:
            metrics = row.get('metrics', {})
            development_rows.append(dict(stage=stage, model=LABELS.get(stage, stage.replace('_', ' ')),
                step=row['step'], epoch=row.get('epoch', ''), weight_source=row.get('weight_source', 'raw'),
                purpose=row.get('purpose', ''), rmse_mm=metrics.get('point', metrics.get('station_macro_rmse_mm', metrics.get('rmse_macro_mm', float('nan')))),
                es_mm=metrics.get('joint', metrics.get('group_ES_mm', metrics.get('es_mm', float('nan'))))))
    result['tables']['training_history'] = _csv(output/'training_history.csv', training_rows,
        ['stage', 'model', 'step', 'epoch', 'metric', 'value'])
    result['tables']['development_history'] = _csv(output/'development_history.csv', development_rows,
        ['stage', 'model', 'step', 'epoch', 'weight_source', 'purpose', 'rmse_mm', 'es_mm'])
    if plotted:
        fig = figure(figsize=(12, 3.0*math.ceil(len(plotted)/3)), constrained_layout=True)
        axes = fig.subplots(math.ceil(len(plotted)/3), 3, squeeze=False).ravel()
        for ax, (stage, history, _) in zip(axes, plotted):
            points = [r for r in history if 'loss' in r and math.isfinite(r['loss'])]
            ax.plot([r['step'] for r in points], [r['loss'] for r in points], linewidth=1)
            ax.set(title=LABELS.get(stage, stage.replace('_', ' ')), xlabel='Optimizer update', ylabel='Training objective')
            ax.grid(alpha=.2)
        for ax in axes[len(plotted):]:
            ax.set_visible(False)
        _save(fig, output/'training_losses.png', result['figures'], 'training_losses')
        fig = figure(figsize=(11, 2.6*len(plotted)), constrained_layout=True)
        axes = fig.subplots(len(plotted), 2, squeeze=False)
        for pair, (stage, _, _) in zip(axes, plotted):
            for ax, metric, title in zip(pair, ('rmse_mm', 'es_mm'), ('Development station RMSE (mm)', 'Development group ES (mm)')):
                for source, style in (('raw', '-'), ('ema', '--')):
                    rows = [r for r in development_rows if r['stage'] == stage and r['weight_source'] == source]
                    if rows:
                        ax.plot([r['step'] for r in rows], [r[metric] for r in rows], style, label=source.upper())
                ax.set(title=LABELS.get(stage, stage.replace('_', ' '))+' | '+title, xlabel='Optimizer update')
                ax.grid(alpha=.2)
                if ax.lines:
                    ax.legend(fontsize=8)
        _save(fig, output/'development_rmse_es.png', result['figures'], 'development_rmse_es')
    else:
        result['skipped'].append(dict(panel='training/development curves', reason='No archived training or development histories.'))

    stations = []; calibrations = []; differences = []; joint_curves = []; point_curves = []; bootstrap_rows = []
    bootstrap_seconds = 0.
    reports = final.get('reports', {})
    for stage in FINAL_STAGES:
        report = reports.get(stage)
        if not report:
            result['skipped'].append(dict(stage=stage, reason='No final selected products in evaluation reports.'))
            continue
        # One loaded archive serves both products when they select the same hash.
        selected = {}
        for product in ('point', 'joint'):
            winner = report.get('products', {}).get(product)
            if not winner:
                result['skipped'].append(dict(stage=stage, product=product, reason='No selected product.'))
                continue
            digest = winner['sha256']
            if not any(row.get('sha256') == digest for row in report.get('candidates', [])):
                raise ValueError('Selected product is absent from final evaluated candidates: '+stage)
            selected.setdefault(digest, []).append(product)
        for digest, products in selected.items():
            archive = Path(campaign.root)/'evaluation'/stage/digest[:20]/'station_predictions.npz'
            reduced = _reduce_archive(archive, bootstrap='point' in products)
            bootstrap_seconds += reduced['bootstrap_seconds']
            label = LABELS[stage]
            result['archives'].append(dict(stage=stage, model=label, sha256=digest, products=products,
                                           path=str(archive.resolve()), members=reduced['members']))
            if not reduced['has_heights']:
                result['skipped'].append(dict(stage=stage, products=products,
                    panel='Height difference <=100 m station-pair diagnostics', reason='Station heights are absent from the archived arrays.'))
            for product in products:
                identity = dict(stage=stage, model=label, product=product, sha256=digest)
                values = reduced['stations']
                for station in range(len(reduced['xy'])):
                    stations.append(dict(**identity, station_index=station,
                        x_m=float(reduced['xy'][station, 0]), y_m=float(reduced['xy'][station, 1]),
                        **{key: (int(array[station]) if key == 'valid_hours' else float(array[station])) for key, array in values.items()}))
                calibrations.extend(dict(**identity, **row) for row in reduced['calibration'])
                differences.extend(dict(**identity, **row) for row in reduced['differences'])
                if product == 'joint':
                    joint_curves.append((label, reduced['calibration']))
                if product != 'point':
                    continue
                point_curves.append((label, reduced['differences']))
                bootstrap_rows.extend(dict(**identity, **row) for row in reduced['bootstrap'])
                fig = figure(figsize=(10.8, 4.4), constrained_layout=True)
                axes = fig.subplots(1, 2)
                use = values['valid_hours'] > 0
                for ax, key, title, cmap in zip(axes, ('rmse_mm', 'rmse_gain_mm'),
                        ('Station RMSE (mm)', 'RMSE gain vs fused background (mm)'), ('viridis', 'RdBu')):
                    kwargs = {}
                    if key == 'rmse_gain_mm':
                        bound = max(float(np.nanmax(abs(values[key][use]))), 1e-6)
                        kwargs = dict(vmin=-bound, vmax=bound)
                    scatter = ax.scatter(reduced['xy'][use, 0]/1000, reduced['xy'][use, 1]/1000,
                        c=values[key][use], cmap=cmap, s=32, edgecolors='white', linewidths=.3, **kwargs)
                    ax.set(title=title, xlabel='Projected X (km)', ylabel='Projected Y (km)', aspect='equal')
                    # Preserve geographic aspect without collapsing a small or
                    # collinear station cohort into an unreadable axis line.
                    coordinates = reduced['xy'][use]/1000
                    lower, upper = coordinates.min(0), coordinates.max(0)
                    span = np.maximum(upper-lower, max(float((upper-lower).max())*.4, 1.))
                    center = (lower+upper)/2
                    ax.set_xlim(center[0]-.56*span[0], center[0]+.56*span[0])
                    ax.set_ylim(center[1]-.56*span[1], center[1]+.56*span[1])
                    fig.colorbar(scatter, ax=ax, shrink=.8)
                fig.suptitle(label+' | independent C stations | point product')
                _save(fig, output/(stage+'_station_accuracy_gain.png'), result['figures'], stage+'_station_accuracy_gain')
            del reduced
    identity_fields = ['stage', 'model', 'product', 'sha256']
    result['tables']['station_accuracy'] = _csv(output/'station_accuracy_gain.csv', stations,
        identity_fields+['station_index', 'x_m', 'y_m', 'valid_hours', 'rmse_mm', 'mae_mm', 'bias_mm', 'background_rmse_mm', 'rmse_gain_mm'])
    result['tables']['calibration'] = _csv(output/'calibration_width.csv', calibrations,
        identity_fields+['level', 'nominal_coverage', 'coverage', 'width_mm', 'scored_stations', 'valid_station_hours'])
    result['tables']['station_differences'] = _csv(output/'station_difference_errors.csv', differences,
        identity_fields+['distance_min_km', 'distance_max_km', 'height_filter', 'geometric_pair_count', 'pair_count', 'valid_pair_hours',
                         'difference_rmse_mm', 'difference_mae_mm', 'difference_bias_mm', 'background_difference_rmse_mm', 'difference_rmse_gain_mm'])
    result['tables']['paired_block_bootstrap'] = _csv(output/'paired_block_bootstrap.csv', bootstrap_rows,
        identity_fields+['block_days', 'actual_block_hours', 'resampling', 'metric', 'rmse_gain_mm',
                         'gain_ci95_low_mm', 'gain_ci95_high_mm', 'requested_replicates', 'valid_replicates',
                         'valid_station_hours', 'scored_stations', 'spatial_groups_50km', 'seed'])
    bootstrap_details = output/'paired_block_bootstrap.json'
    bootstrap_details.write_text(json.dumps(dict(seed=20261122, requested_replicates=2000,
        seed_derivation='Time RNG: seed + block_days; independent group RNG: seed + 1000 + block_days. Same streams reused across model archives.',
        block_days=[7, 14], confidence_level=.95, ci_method='Paired percentile interval',
        time_resampling='Uniform circular moving-block starts, synchronized across stations and paired predictors; final block truncated to original calendar length.',
        spatial_group_width_km=50, spatial_groups='Fixed projected 50-km square cells, origin (0 m, 0 m); groups drawn uniformly with replacement, including all their scored stations.',
        grouping_interpretation='Spatial group resampling sensitivity; cells are not selected using model errors or described as an independent dense truth cohort.',
        short_record_policy='Block duration is capped at archived calendar duration and reported as actual_block_hours.',
        metrics=['station_macro_rmse', 'micro_rmse'], gain_sign='Background RMSE minus model RMSE; positive is improvement.',
        cpu_seconds=bootstrap_seconds, rows=bootstrap_rows), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    result['bootstrap_details'] = str(bootstrap_details.resolve())
    if joint_curves:
        fig = figure(figsize=(12, 3.8), constrained_layout=True)
        axes = fig.subplots(1, 3)
        for label, rows in joint_curves:
            nominal = [row['nominal_coverage'] for row in rows]
            coverage = [row['coverage'] for row in rows]; width = [row['width_mm'] for row in rows]
            axes[0].plot(nominal, coverage, 'o-', label=label)
            axes[1].plot(nominal, width, 'o-', label=label)
            axes[2].plot(width, coverage, 'o-', label=label)
            for x, y, level in zip(width, coverage, (80, 90, 95)):
                axes[2].annotate(str(level), (x, y), fontsize=7, xytext=(3, 3), textcoords='offset points')
        axes[0].plot([.8, .95], [.8, .95], 'k--', linewidth=.8)
        axes[0].set(xlabel='Nominal coverage', ylabel='Observed station-macro coverage', ylim=(0, 1.02))
        axes[1].set(xlabel='Nominal coverage', ylabel='Mean interval width (mm)')
        axes[2].set(xlabel='Mean interval width (mm)', ylabel='Observed station-macro coverage', ylim=(0, 1.02))
        for ax in axes:
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle('Independent C stations | joint products | observation prediction intervals')
        _save(fig, output/'calibration_width.png', result['figures'], 'calibration_width')
    if point_curves:
        fig = figure(figsize=(11, 4.1), constrained_layout=True)
        axes = fig.subplots(1, 2)
        for ax, filter_name, title in zip(axes, ('all', 'difference_le_100m'), ('All station pairs', 'Height difference <=100 m')):
            for label, rows in point_curves:
                group = [r for r in rows if r['height_filter'] == filter_name]
                if group:
                    ax.plot(range(4), [r['difference_rmse_mm'] for r in group], 'o-', label=label)
            ax.set(title=title, xlabel='Station separation (km)', ylabel='Station-difference RMSE (mm)')
            ax.set_xticks(range(4), ['0-25', '25-50', '50-100', '100-200'])
            ax.grid(alpha=.2)
            if not ax.lines:
                ax.text(.5, .5, 'Station heights not archived', transform=ax.transAxes, ha='center')
        axes[0].legend(fontsize=7)
        fig.suptitle('Independent C stations | point products | counts in accompanying CSV')
        _save(fig, output/'station_difference_errors.png', result['figures'], 'station_difference_errors')
    manifest = output/'figure_manifest.json'
    result['manifest'] = str(manifest.resolve())
    result['cpu_export_seconds'] = time.monotonic()-export_started
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result
