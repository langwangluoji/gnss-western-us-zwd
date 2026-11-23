"""Fixed-support station evaluation; target availability is used only for scoring."""
from pathlib import Path
import json
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F


def sample_stations(field, xy, factor):
    """Interpolate common-height millimetres, then restore station height."""
    xy = torch.as_tensor(xy, dtype=field.dtype, device=field.device)
    factor = torch.as_tensor(factor, dtype=field.dtype, device=field.device)
    grid = xy[None, :, None].expand(len(field), -1, -1, -1)
    values = F.grid_sample(field, grid, mode='bilinear', padding_mode='zeros',
                           align_corners=False)[:, 0, :, 0]
    return values * factor


def station_support(mask, xy):
    """Require every nonzero interpolation weight to have supported grid data."""
    mask = mask.float()
    xy = torch.as_tensor(xy, dtype=mask.dtype, device=mask.device)
    sampled = sample_stations(mask, xy, torch.ones(len(xy), device=mask.device))
    return (sampled >= 1 - 1e-6) & (xy.abs() <= 1).all(-1)[None]


def point_metrics(prediction, truth, valid):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    valid = np.asarray(valid, bool)
    if prediction.shape != truth.shape or valid.shape != truth.shape:
        raise ValueError('Prediction, truth and eligibility shapes must agree')
    if not np.isfinite(prediction[valid]).all() or not np.isfinite(truth[valid]).all():
        raise ValueError('Non-finite value on the fixed evaluation support')
    errors = np.where(valid, prediction - truth, 0.)
    count = valid.sum(0)
    scored = count > 0
    total = int(count.sum())
    if not total:
        return dict(station_macro_rmse_mm=None, station_rmse_mm=None,
                    station_mse_mm2=None, station_mae_mm=None, station_bias_mm=None,
                    stations_scored=0, station_records=0, coverage_fraction=0.)
    sse = np.square(errors).sum(0)
    return dict(station_macro_rmse_mm=float(np.sqrt(sse[scored] / count[scored]).mean()),
                station_rmse_mm=float(np.sqrt(sse.sum() / total)),
                station_mse_mm2=float(sse.sum() / total),
                station_mae_mm=float(np.abs(errors).sum() / total),
                station_bias_mm=float(errors.sum() / total),
                stations_scored=int(scored.sum()), station_records=total,
                coverage_fraction=float(total / valid.size))


def probability_metrics(samples, truth, valid):
    """Empirical ensemble CRPS and joint scores, all in surface millimetres.

    samples: [time, member, station]. Energy is normalized by sqrt(station count);
    variogram uses p=0.5 and the mean over unordered valid station pairs.
    """
    samples = np.asarray(samples, dtype=np.float64)
    truth, valid = np.asarray(truth), np.asarray(valid, bool)
    if samples.ndim != 3 or samples.shape[::2] != truth.shape or valid.shape != truth.shape:
        raise ValueError('Expected samples [time,member,station] and matching observations')
    if samples.shape[1] < 1:
        raise ValueError('At least one ensemble member is required')
    if not np.isfinite(samples.transpose(0, 2, 1)[valid]).all():
        raise ValueError('Non-finite ensemble on the fixed evaluation support')
    if not np.isfinite(truth[valid]).all():
        raise ValueError('Non-finite observation on the fixed evaluation support')
    k = samples.shape[1]
    ordered = np.sort(samples, axis=1)
    coefficient = (2 * np.arange(1, k + 1) - k - 1)[None, :, None]
    pair_half = (ordered * coefficient).sum(1) / (k * k)
    crps = np.abs(samples - truth[:, None]).mean(1) - pair_half
    result = dict(crps_mm=float(crps[valid].mean()) if valid.any() else None,
                  ensemble_members=k, crps_estimator='empirical ensemble (biased finite-K)',
                  energy_normalization='Euclidean norm / sqrt(number of valid stations)',
                  variogram_p=.5)
    for level in (80, 90, 95):
        tail = (1 - level / 100) / 2
        low, high = np.quantile(samples, [tail, 1 - tail], axis=1)
        result[f'coverage_{level}'] = (float(((truth >= low) & (truth <= high))[valid].mean())
                                      if valid.any() else None)
        result[f'width_{level}_mm'] = float((high - low)[valid].mean()) if valid.any() else None
    energies, variograms = [], []
    for t in range(len(truth)):
        chosen = np.flatnonzero(valid[t])
        if not len(chosen):
            continue
        values = samples[t][:, chosen]
        observed = truth[t, chosen]
        first = np.linalg.norm(values - observed, axis=1).mean()
        # Avoid a [K,K,N] allocation for large station sets.
        pair_sum = sum(np.linalg.norm(values - row, axis=1).sum() for row in values)
        energies.append((first - pair_sum / (2 * k * k)) / np.sqrt(len(chosen)))
        if len(chosen) > 1:
            i, j = np.triu_indices(len(chosen), 1)
            observed_difference = np.abs(observed[i] - observed[j]) ** .5
            predicted_difference = (np.abs(values[:, i] - values[:, j]) ** .5).mean(0)
            variograms.append(np.square(observed_difference - predicted_difference).mean())
    result['energy_score_mm'] = float(np.mean(energies)) if energies else None
    result['variogram_score_mm'] = float(np.mean(variograms)) if variograms else None
    result['joint_scored_times'] = len(energies)
    return result


def paired_block_bootstrap(errors_a, errors_b, valid, hours, repeats=2000, seed=20261113):
    """Delta is macroRMSE(A)-macroRMSE(B); one seven-day draw is shared by all stations."""
    a, b, valid = np.asarray(errors_a), np.asarray(errors_b), np.asarray(valid, bool)
    hours = np.asarray(hours, dtype=np.int64)
    if a.shape != b.shape or a.shape != valid.shape or len(hours) != len(a):
        raise ValueError('Bootstrap inputs must share time and station axes')
    if not valid.any() or repeats < 1:
        raise ValueError('Bootstrap requires observations and positive repeats')
    if not np.isfinite(a[valid]).all() or not np.isfinite(b[valid]).all():
        raise ValueError('Non-finite paired errors')
    blocks = (hours - hours.min()) // (7 * 24)
    unique = np.unique(blocks)
    count = np.stack([valid[blocks == block].sum(0) for block in unique])
    sse_a = np.stack([np.where(valid[blocks == block], a[blocks == block] ** 2, 0).sum(0)
                      for block in unique])
    sse_b = np.stack([np.where(valid[blocks == block], b[blocks == block] ** 2, 0).sum(0)
                      for block in unique])

    def macro(sse, counts):
        present = counts > 0
        return np.sqrt(sse[present] / counts[present]).mean()

    actual = macro(sse_a.sum(0), count.sum(0)) - macro(sse_b.sum(0), count.sum(0))
    rng = np.random.default_rng(seed)
    delta = np.empty(repeats)
    for repeat in range(repeats):
        picked = rng.integers(0, len(unique), size=len(unique))
        counts = count[picked].sum(0)
        delta[repeat] = macro(sse_a[picked].sum(0), counts) - macro(sse_b[picked].sum(0), counts)
    return dict(delta_macro_rmse_mm=float(actual), ci95_mm=np.quantile(delta, [.025, .975]).tolist(),
                repeats=int(repeats), seed=int(seed), block_hours=168, blocks=len(unique),
                delta_definition='macroRMSE(A)-macroRMSE(B)',
                uncertainty='paired time-block sampling; training-seed variation separate')


def evaluate(model, data, indices, members=8, seed=20261113, output=None,
             role='validation', years=(2019,), probability=False):
    """Evaluate without reading training targets; save optional station and grid summaries."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices) or members < 1:
        raise ValueError('Evaluation requires samples and a positive ensemble size')
    reference = data.evaluation(role=role, years=years)
    truth = np.asarray(reference['truth'])[indices].astype(np.float64)
    names = [str(x) for x in reference['names']]
    n = len(names)
    prediction = np.empty((len(indices), n), dtype=np.float64)
    valid = np.zeros_like(prediction, bool)
    all_domain = np.zeros_like(valid)
    ensemble = []
    event_fields, event_hours = [], []
    # Selection depends only on time. An explicitly registered manifest wins.
    manifest = getattr(data, 'manifest', {})
    if 'event_indices' in manifest:
        event_indices = set(map(int, manifest['event_indices']))
    else:
        month = np.asarray(data.hours, dtype='datetime64[h]')[indices].astype('datetime64[M]')
        _, first = np.unique(month, return_index=True)
        event_indices = set(map(int, indices[first]))
    destination = None
    handle = None
    if output is not None:
        destination = Path(output)
        if destination.suffix.lower() not in ('.h5', '.hdf5'):
            destination = destination / 'evaluation.h5'
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle = h5py.File(destination.with_suffix('.partial.h5'), 'w')
        handle.create_dataset('hour_id', data=np.asarray(data.hours)[indices])
        handle.create_dataset('station', data=np.asarray(names, dtype=h5py.string_dtype('utf-8')))
        for name in ('grid_mean_z0_mm', 'grid_std_z0_mm'):
            handle.create_dataset(name, (len(indices), 24, 24), dtype='f4', compression='gzip')
    old_training = model.training if model is not None else None
    if model is not None:
        model.eval()
    arm = getattr(model, 'arm', 'mean')
    device = None
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for offset in range(0, len(indices), 4):
                ids = indices[offset:offset + 4]
                batch = data.batch(ids, targets=False)
                mu = batch['mu_mm']
                if device is None:
                    device = mu.device
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                xy = torch.as_tensor(reference['xy'], device=mu.device, dtype=mu.dtype)
                factor = torch.as_tensor(reference['factor'], device=mu.device, dtype=mu.dtype)
                domain = batch['domain'].to(mu.device).expand(len(ids), -1, -1, -1)
                supported = station_support(domain, xy)
                context = station_support(batch['context_support'].to(mu.device), xy)
                section = slice(offset, offset + len(ids))
                observed = np.isfinite(truth[section])
                all_domain[section] = supported.cpu().numpy() & observed
                valid[section] = (supported & context).cpu().numpy() & observed
                if model is None:
                    fields = mu[:, None]
                    mean_field = mu
                elif arm in ('det', 'gaussian'):
                    forward = model(batch)
                    residual_mean = forward[0] if arm == 'gaussian' else forward
                    mean_field = mu + float(data.scale) * residual_mean
                    if probability and arm == 'gaussian':
                        condition = model.encode_condition(batch)
                        residual = model.sample(condition, np.asarray(data.hours)[ids], members, seed)
                        fields = mu[:, None] + float(data.scale) * residual
                    else:
                        fields = mean_field[:, None]
                else:
                    condition = model.encode_condition(batch)
                    residual = model.sample(condition, np.asarray(data.hours)[ids], members, seed)
                    fields = mu[:, None] + float(data.scale) * residual
                    mean_field = fields.mean(1)
                station_mean = sample_stations(mean_field, xy, factor).cpu().numpy()
                # Check the expanded support too, so failures never disappear from a secondary table.
                if not np.isfinite(station_mean[all_domain[section]]).all():
                    raise ValueError('Non-finite prediction on the fixed evaluation support')
                prediction[section] = station_mean
                if probability:
                    b, k = fields.shape[:2]
                    sampled = sample_stations(fields.flatten(0, 1), xy, factor).reshape(b, k, n)
                    ensemble.append(sampled.cpu().numpy())
                if handle is not None:
                    mask = domain[:, 0].bool()
                    grid_mean = torch.where(mask, mean_field[:, 0], float('nan'))
                    grid_std = torch.where(mask, fields.std(1, unbiased=False)[:, 0], float('nan'))
                    handle['grid_mean_z0_mm'][section] = grid_mean.cpu().numpy()
                    handle['grid_std_z0_mm'][section] = grid_std.cpu().numpy()
                    for local, index in enumerate(ids):
                        if int(index) in event_indices:
                            event_fields.append(torch.where(domain[local].bool(), fields[local],
                                                            float('nan'))[:, 0].cpu().numpy())
                            event_hours.append(int(data.hours[index]))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        result = point_metrics(prediction, truth, valid)
        if result['station_records'] == 0:
            raise ValueError('No observations on the fixed primary evaluation support')
        result.update(all_domain=point_metrics(prediction, truth, all_domain), arm=arm,
                      role=role, years=list(years), hours=len(indices), requested_members=int(members),
                      inference_seconds=float(elapsed), nfe_per_member=35 if arm == 'edm' else (0 if arm == 'mean' else 1),
                      sample_level_nfe=int(len(indices) * (35 * members if arm == 'edm' else
                                          (0 if arm == 'mean' else (2 if arm == 'gaussian' and probability else 1)))),
                      peak_memory_bytes=(int(torch.cuda.max_memory_allocated(device))
                                         if device.type == 'cuda' else 0),
                      support_rule='fixed D interpolation stencil AND current context support AND finite observation',
                      timing_scope='batch preparation, residual inference, interpolation and optional artifact writes; cached mean preparation excluded',
                      nominal_time_basis='source hour_id; no UTC/GPST conversion')
        # Reference-label coverage is a diagnostic stratum, never a model input or eligibility rule.
        complete = getattr(data, 'teacher_complete', None)
        if complete is not None:
            complete = np.asarray(complete, dtype=bool)[indices]
            result['teacher_complete_group'] = point_metrics(prediction, truth, valid & complete[:, None])
            result['teacher_incomplete_group'] = point_metrics(prediction, truth, valid & ~complete[:, None])
        if probability:
            samples = np.concatenate(ensemble)
            result['probability'] = probability_metrics(samples, truth, valid)
            result['all_domain_probability'] = probability_metrics(samples, truth, all_domain)
        if handle is not None:
            for key, value in dict(prediction_mm=prediction, truth_mm=truth, valid=valid,
                                   all_domain_valid=all_domain).items():
                handle.create_dataset(key, data=value, compression='gzip')
            if probability:
                handle.create_dataset('station_samples_mm', data=samples, compression='gzip')
            if event_fields:
                group = handle.create_group('event_samples')
                group.create_dataset('hour_id', data=event_hours)
                group.create_dataset('z0_mm', data=np.stack(event_fields), compression='gzip')
                group.create_dataset('domain', data=batch['domain'][0, 0].cpu().numpy())
                group.attrs['selection'] = ('manifest event_indices' if 'event_indices' in manifest else
                                            'first eligible evaluated nominal hour in each month; independent of label values')
            handle.attrs['metrics_json'] = json.dumps(result, ensure_ascii=False)
            handle.attrs['complete'] = True
            handle.close()
            handle = None
            destination.with_suffix('.partial.h5').replace(destination)
            destination.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result
    finally:
        if handle is not None:
            handle.close()
        if model is not None:
            model.train(old_training)
