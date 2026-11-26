"""Unrounded product-specific checkpoint selection and retrospective access lock."""
import copy
import re

import numpy as np

from ..artifacts import fingerprint


PRODUCT_METRICS = dict(point='station_macro_rmse_mm', joint='group_macro_es_mm',
                       marginal='station_macro_crps_mm')


def selection_values(bundle):
    result = {'point': {'value': bundle['primary']['station_macro_rmse_mm']}}
    if 'station_macro_crps_mm' in bundle['primary']:
        result['marginal'] = {'value': bundle['primary']['station_macro_crps_mm']}
    if bundle.get('joint', {}).get('group_macro_es_mm') is not None:
        result['joint'] = {'value': bundle['joint']['group_macro_es_mm'],
                           'vs': bundle['joint']['group_macro_vs_mm']}
    return result


def _metric(record, product):
    metric = PRODUCT_METRICS[product]
    value = record.get(metric, record.get('metrics', {}).get(metric))
    if value is None and product in record.get('selection_values', {}):
        value = record['selection_values'][product]['value']
    return value


def keep_top_candidates(records, products=('point', 'joint', 'marginal'), top=3, *, require_full_year=False):
    """Keep separate top-three screens or full-year reranks for each product."""
    if top < 1 or set(products)-set(PRODUCT_METRICS):
        raise ValueError('Positive top and registered product names required')
    eligible = [r for r in records if 'pretrain' not in str(r.get('stage', '')).lower()
                and int(r.get('step', 0)) > 200]
    if require_full_year:
        eligible = [r for r in eligible if r.get('year') == 2019 and r.get('full_year') is True]
    result = {}
    for product in products:
        candidates = []
        for row in eligible:
            value = _metric(row, product)
            if value is not None:
                if not np.isfinite(value):
                    raise ValueError('Checkpoint selection requires finite unrounded metrics')
                candidates.append(row)
        candidates.sort(key=lambda r: (float(_metric(r, product)), int(r['step']), str(r.get('checkpoint', ''))))
        result[product] = copy.deepcopy(candidates[:top])
    return result


def select_products(records, *, products=('point', 'joint', 'marginal'), top=3, require_full_year=False):
    candidates = keep_top_candidates(records, products, top, require_full_year=require_full_year)
    return {product: dict(selected=rows[0] if rows else None, top_candidates=rows,
                          metric=PRODUCT_METRICS[product], full_year_required=require_full_year)
            for product, rows in candidates.items()}


def select_bstar(records, registered_controls, *, fixed_controls=('coarse_bilinear', 'A_IDW', 'A_kriging')):
    """Lock Bstar from the preregistered point-control names, on full 2019 only."""
    registered = set(registered_controls)
    fixed = set(fixed_controls)
    candidates = [r for r in records if r.get('model') in registered and
                  r.get('year') == 2019 and r.get('full_year') is True and
                  'pretrain' not in str(r.get('stage', '')).lower() and
                  (int(r.get('step', 0)) > 200 or
                   (r.get('model') in fixed and int(r.get('step', 0)) == 0)) and
                  _metric(r, 'point') is not None]
    if not candidates:
        raise ValueError('Bstar requires registered point controls with full 2019 metrics')
    if not all(np.isfinite(_metric(r, 'point')) for r in candidates):
        raise ValueError('Bstar requires finite unrounded point metrics')
    candidates.sort(key=lambda r: (float(_metric(r, 'point')), int(r.get('step', 0)), str(r.get('checkpoint', ''))))
    return copy.deepcopy(candidates[0])


def sampling_stability(samples64, samples128, *, point64_mm, point128_mm, order64, order128,
                       gates64, gates128, tolerance_mm=.005, member_ids64=None, member_ids128=None,
                       prefix_tolerance_mm=.001, noise_provenance64=None, noise_provenance128=None):
    """Check fixed-noise/member provenance, FP32 prefix tolerance and metric stability."""
    a, b = np.asarray(samples64), np.asarray(samples128)
    if a.ndim != 3 or b.ndim != 3 or a.shape[1] != 64 or b.shape[1] != 128 or a.shape[::2] != b.shape[::2]:
        raise ValueError('Expected aligned [T,64,N] and [T,128,N] member arrays')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite([point64_mm, point128_mm]).all():
        raise ValueError('Sampling-stability inputs must be finite')
    if not np.isfinite([tolerance_mm, prefix_tolerance_mm]).all() or min(tolerance_mm, prefix_tolerance_mm) < 0:
        raise ValueError('Stability tolerance must be nonnegative')
    if prefix_tolerance_mm > .001:
        raise ValueError('Forecast prefix tolerance must be at most 0.001 mm')
    member_ids_match = False
    if member_ids64 is not None or member_ids128 is not None:
        ids64, ids128 = np.asarray(member_ids64), np.asarray(member_ids128)
        if (ids64.shape != (64,) or ids128.shape != (128,) or
                len(np.unique(ids128)) != 128 or not np.array_equal(ids64, ids128[:64])):
            raise ValueError('Global member IDs must be unique with an identical 64-member prefix')
        member_ids_match = True
    noise_matches = (noise_provenance64 is not None and noise_provenance128 is not None and
                     bool(noise_provenance64) and bool(noise_provenance128) and
                     fingerprint(noise_provenance64) == fingerprint(noise_provenance128))
    provenance_matches = member_ids_match and noise_matches
    prefix_delta = float(np.max(abs(a.astype(np.float64)-b[:, :64].astype(np.float64))))
    prefix = prefix_delta <= prefix_tolerance_mm
    delta = float(abs(point128_mm-point64_mm))
    order_changed = list(order64) != list(order128)
    gates_changed = gates64 != gates128
    stable = provenance_matches and prefix and delta <= tolerance_mm and not order_changed and not gates_changed
    return dict(stable=bool(stable), prefix_matches=bool(prefix), point_delta_mm=delta,
                provenance_matches=bool(provenance_matches), member_ids_match=member_ids_match,
                noise_provenance_matches=bool(noise_matches),
                noise_provenance_sha256=fingerprint(noise_provenance64) if noise_matches else None,
                prefix_max_abs_difference_mm=prefix_delta, prefix_tolerance_mm=prefix_tolerance_mm,
                tolerance_mm=tolerance_mm, order_changed=order_changed, gates_changed=gates_changed,
                members=64 if stable else 128, requires_full_128_rerank=not stable,
                stability_year=2019)


def _hash(value, name):
    if not isinstance(value, str) or re.fullmatch('[0-9a-fA-F]{64}', value) is None:
        raise ValueError(f'{name} must be a SHA256 hexadecimal digest')
    return value.lower()


def build_product_lock(products, *, config_sha256, manifest_sha256, stability, protocol_sha256=None):
    """Build JSON-compatible immutable lock metadata, with no data/checkpoint I/O."""
    config_sha256 = _hash(config_sha256, 'config_sha256')
    manifest_sha256 = _hash(manifest_sha256, 'manifest_sha256')
    if not stability.get('stable') and not stability.get('full_128_rerank_complete'):
        raise ValueError('Lock requires stable K64/K128 or completed uniform K128 rerank')
    locked = {}
    for product, entry in products.items():
        selected = entry.get('selected')
        if selected is None:
            continue
        if selected.get('year') != 2019 or selected.get('full_year') is not True:
            raise ValueError('Lock requires full-year 2019 rerank for each selected product')
        if 'pretrain' in str(selected.get('stage', '')).lower() or int(selected.get('step', 0)) <= 200:
            raise ValueError('Final pretraining/startup checkpoints cannot be locked')
        checkpoint_sha256 = _hash(selected.get('checkpoint_sha256'), 'checkpoint_sha256')
        if _metric(selected, product) is None:
            raise ValueError('Lock requires selected product metric')
        locked[product] = copy.deepcopy(selected)
        locked[product]['checkpoint_sha256'] = checkpoint_sha256
    if not locked:
        raise ValueError('At least one selected product is required')
    result = dict(schema='gnss_field_product_lock_v1', selection_year=2019,
                  retrospective_year=2020, retrospective_role='D', config_sha256=config_sha256,
                  manifest_sha256=manifest_sha256, products=locked, stability=copy.deepcopy(stability))
    if protocol_sha256 is not None:
        result['protocol_sha256'] = _hash(protocol_sha256, 'protocol_sha256')
    result['lock_sha256'] = fingerprint(result)
    return result


def validate_retrospective_access(lock, *, year, role, config_sha256, manifest_sha256,
                                  checkpoint_sha256, product=None):
    """Authorize D2020 only for content-addressed locked 2019 products."""
    payload = {k: v for k, v in lock.items() if k != 'lock_sha256'}
    if lock.get('lock_sha256') != fingerprint(payload):
        raise ValueError('Product lock integrity mismatch')
    if lock.get('selection_year') != 2019 or year != 2020 or role != 'D':
        raise ValueError('Retrospective access requires 2019 selection and D2020 role')
    if _hash(config_sha256, 'config_sha256') != lock.get('config_sha256') or _hash(manifest_sha256, 'manifest_sha256') != lock.get('manifest_sha256'):
        raise ValueError('Retrospective configuration/manifest changed after lock')
    products = lock.get('products', {})
    allowed = [products[product]] if product in products else list(products.values())
    if product is not None and product not in products:
        raise ValueError('Requested product is absent from lock')
    checkpoint = _hash(checkpoint_sha256, 'checkpoint_sha256')
    if not any(r.get('checkpoint_sha256') == checkpoint for r in allowed):
        raise ValueError('Retrospective checkpoint is absent from product lock')
    return True


def direction_gate(comparisons, *, formal=True):
    """Four preregistered fine endpoints as an intersection-union decision."""
    if len(comparisons) != 4:
        raise ValueError('The fine gate requires exactly four preregistered comparisons')
    decisions = {}
    for name, comparison in comparisons.items():
        difference = comparison.get('difference_mm', comparison.get('difference'))
        if difference is None or not np.isfinite(difference):
            raise ValueError('Comparison differences must be finite')
        ci = comparison.get('ci95_mm', comparison.get('ci95'))
        if formal:
            if ci is None or len(ci) != 2 or not np.isfinite(ci).all() or ci[0] > ci[1]:
                raise ValueError('Formal gate requires finite ordered paired 95 percent CI')
            decisions[name] = bool(ci[1] < 0)
        else:
            decisions[name] = bool(difference < 0)
    return dict(fine_pass=all(decisions.values()), endpoints=decisions,
                mode='formal_paired_CI' if formal else 'pilot_direction', family='fine')


def probability_gate(*, es_difference, vs_difference, crps_difference, intervals,
                      coverage_difference_ci=None, formal=False):
    """Separate marginal, joint and calibration flags without a composite score."""
    values = [es_difference, vs_difference, crps_difference]
    if not np.isfinite(values).all():
        raise ValueError('Probability differences must be finite')
    engineering = {}
    confirmation = {}
    for level in (.8, .9, .95):
        key = str(round(level*100))
        if key not in intervals or intervals[key].get('coverage') is None:
            raise ValueError('All registered interval levels are required')
        coverage = intervals[key]['coverage']
        engineering[key] = bool(abs(coverage-level) <= .03)
        ci = None if coverage_difference_ci is None else coverage_difference_ci.get(key)
        if ci is not None and (len(ci) != 2 or not np.isfinite(ci).all() or ci[0] > ci[1]):
            raise ValueError('Coverage-difference confidence intervals must be finite and ordered')
        confirmation[key] = bool(ci[0] >= -.03 and ci[1] <= .03) if ci is not None else None
    confirmed = all(v is True for v in confirmation.values())
    calibration = confirmed if formal else all(engineering.values())
    return dict(marginal_improves=bool(crps_difference < 0),
                joint_improves=bool(es_difference < 0 and vs_difference < 0 and crps_difference <= 0),
                engineering_coverage=engineering, engineering_coverage_pass=all(engineering.values()),
                formal_coverage=confirmation,
                formal_coverage_status='confirmed' if confirmed else 'indeterminate',
                calibrated_joint_pass=bool(es_difference < 0 and vs_difference < 0 and crps_difference <= 0 and calibration))


def holm_adjust(p_values):
    """Holm-adjust an explicitly preregistered family of selected-product claims."""
    values = np.asarray(p_values, float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('p values must be a finite vector between zero and one')
    order = np.argsort(values, kind='stable')
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    return adjusted
