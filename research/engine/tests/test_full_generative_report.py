from types import SimpleNamespace
import csv
import json
from pathlib import Path

import numpy as np
import pytest


def _campaign(tmp_path, heights=True):
    stages = {}
    for stage in ('control', 'complete_vae', 'complete_gan', 'complete_diffusion'):
        digest = (stage + 'x'*64)[:64]
        output = tmp_path/'evaluation'/stage/digest[:20]
        output.mkdir(parents=True)
        arrays = dict(hours=np.array([1, 2]), truth=np.zeros((2, 4)),
            mean=np.array([[1., 3., 5., 7.]]*2), background_mean=np.full((2, 4), 4.),
            population=np.array([[True]*4, [False, True, True, True]]),
            samples=np.broadcast_to(np.array([-2., 0., 2.])[None, :, None], (2, 3, 4)),
            xy=np.array([[0., 0.], [20000., 0.], [60000., 0.], [160000., 0.]]))
        # Station 0 has one scored miss; the other stations have two hits each.
        # Station-macro coverage is 3/4, whereas pooled coverage would be 6/7.
        arrays['samples'] = arrays['samples'].copy()
        arrays['samples'][:, :, 0] += 12.
        if heights:
            arrays['heights'] = np.array([0., 50., 200., 20.])
        np.savez_compressed(output/'station_predictions.npz', **arrays)
        stages[stage] = dict(status='complete', history=[dict(step=1, loss=9.), dict(step=2, loss=3.)],
            evaluation_history=[dict(step=0, weight_source='raw', metrics=dict(point=4., joint=5.)),
                                dict(step=2, weight_source='ema', metrics=dict(point=2., joint=3.))])
        stages.setdefault('final_evaluation', dict(status='complete', role='C', year=2019, reports={}))
        stages['final_evaluation']['reports'][stage] = dict(
            candidates=[dict(sha256=digest, checkpoint='unused.pt')],
            products={product: dict(sha256=digest, checkpoint='unused.pt') for product in ('point', 'joint')})
    return SimpleNamespace(root=tmp_path, manifest=dict(stages=stages))


def _rows(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def test_exports_selected_products_masked_station_gains_and_macro_intervals(tmp_path):
    # Catches absent files, using samples for point accuracy, ignoring masks,
    # and accidentally reporting pooled rather than per-station summaries.
    from gnss_field.generative.report import export_campaign_figures
    result = export_campaign_figures(_campaign(tmp_path))
    assert len(result['archives']) == 4  # identical point/joint checkpoint is read once
    assert {r['stage'] for r in result['archives']} == {'control', 'complete_vae', 'complete_gan', 'complete_diffusion'}
    stations = _rows(result['tables']['station_accuracy'])
    vae = [r for r in stations if r['stage']=='complete_vae' and r['product']=='point']
    assert [int(r['valid_hours']) for r in vae] == [1, 2, 2, 2]
    assert [float(r['rmse_mm']) for r in vae] == [1., 3., 5., 7.]
    assert [float(r['rmse_gain_mm']) for r in vae] == [3., 1., -1., -3.]
    intervals = [r for r in _rows(result['tables']['calibration']) if r['stage']=='complete_vae' and r['product']=='joint']
    assert [float(r['coverage']) for r in intervals] == [.75, .75, .75]
    assert [float(r['width_mm']) for r in intervals] == pytest.approx([3.2, 3.6, 3.8])
    assert [float(r['nominal_coverage']) for r in intervals] == [.8, .9, .95]
    for path in result['figures'].values():
        assert Path(path).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert json.loads(Path(result['manifest']).read_text(encoding='utf-8'))['evaluation_role'] == 'C'


def test_distance_bins_count_actual_valid_pairs_and_height_filter(tmp_path):
    # Catches bin-boundary overlap, counting geometric pairs as observations,
    # and applying the <=100m elevation filter in the wrong units.
    from gnss_field.generative.report import export_campaign_figures
    result = export_campaign_figures(_campaign(tmp_path))
    rows = [r for r in _rows(result['tables']['station_differences']) if r['stage']=='complete_vae' and r['product']=='point']
    all_height = [r for r in rows if r['height_filter']=='all']
    assert [int(r['pair_count']) for r in all_height] == [1, 1, 1, 3]
    assert [int(r['valid_pair_hours']) for r in all_height] == [1, 2, 1, 5]
    assert [float(r['difference_rmse_mm']) for r in all_height] == pytest.approx([2., 2., 4., np.sqrt(15.2)])
    low_height = [r for r in rows if r['height_filter']=='difference_le_100m']
    assert [int(r['pair_count']) for r in low_height] == [1, 0, 0, 2]
    assert [int(r['valid_pair_hours']) for r in low_height] == [1, 0, 0, 3]
    assert low_height[1]['difference_rmse_mm'] == ''


def test_missing_heights_is_explicit_and_unselected_archives_are_never_read(tmp_path, monkeypatch):
    from gnss_field.generative.report import export_campaign_figures
    campaign = _campaign(tmp_path, heights=False)
    reports = campaign.manifest['stages']['final_evaluation']['reports']
    # A nonfinal model and an unselected broken archive must not be loaded.
    reports['shared_vae'] = dict(products=dict(point=dict(sha256='broken')))
    reports['complete_vae']['candidates'].append(dict(sha256='broken', checkpoint='broken.pt'))
    real_load = np.load
    loaded = []
    def track_load(path, *args, **kwargs):
        loaded.append(str(path))
        return real_load(path, *args, **kwargs)
    # Observe actual filesystem reads at the expensive archive boundary.
    monkeypatch.setattr(np, 'load', track_load)
    result = export_campaign_figures(campaign)
    assert len(result['archives']) == 4
    assert len(loaded) == 4 and len(set(loaded)) == 4
    assert all(r['height_filter']=='all' for r in _rows(result['tables']['station_differences']))
    assert any('height' in row['reason'].lower() for row in result['skipped'])
    campaign.manifest['stages']['final_evaluation']['status'] = 'paused'
    with pytest.raises(ValueError, match='complete'):
        export_campaign_figures(campaign)


def test_nonfinite_forecasts_fail_even_outside_the_scored_population(tmp_path):
    from gnss_field.generative.report import export_campaign_figures
    campaign = _campaign(tmp_path)
    path = next((tmp_path/'evaluation'/'control').glob('*/station_predictions.npz'))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['samples'][1, 0, 0] = np.nan  # This station-hour has mask=False.
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='Nonfinite'):
        export_campaign_figures(campaign)


def test_point_and_joint_winners_use_their_own_archives(tmp_path):
    # Catches silently reusing point-selected predictions for the joint product.
    from gnss_field.generative.report import export_campaign_figures
    campaign = _campaign(tmp_path)
    report = campaign.manifest['stages']['final_evaluation']['reports']['complete_gan']
    original = next((tmp_path/'evaluation'/'complete_gan').glob('*/station_predictions.npz'))
    with np.load(original, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['mean'] += 10.
    arrays['samples'] *= 2.
    digest = 'distinct_joint_winner'+'z'*43
    output = tmp_path/'evaluation'/'complete_gan'/digest[:20]
    output.mkdir(parents=True)
    np.savez_compressed(output/'station_predictions.npz', **arrays)
    report['products']['joint'] = dict(sha256=digest, checkpoint='joint.pt')
    report['candidates'].append(dict(sha256=digest, checkpoint='joint.pt'))
    result = export_campaign_figures(campaign)
    assert len(result['archives']) == 5
    station_rows = [row for row in _rows(result['tables']['station_accuracy']) if row['stage']=='complete_gan']
    assert [float(row['rmse_mm']) for row in station_rows if row['product']=='point'] == [1., 3., 5., 7.]
    assert [float(row['rmse_mm']) for row in station_rows if row['product']=='joint'] == [11., 13., 15., 17.]
    calibration = [row for row in _rows(result['tables']['calibration']) if row['stage']=='complete_gan' and row['product']=='joint']
    assert [float(row['width_mm']) for row in calibration] == pytest.approx([6.4, 7.2, 7.6])


def test_paired_block_bootstrap_constant_gain_is_positive_and_masked(tmp_path):
    from gnss_field.generative.report import export_campaign_figures
    campaign = _campaign(tmp_path)
    for path in (tmp_path/'evaluation').glob('*/*/station_predictions.npz'):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays['mean'][:] = 1.
        arrays['background_mean'][:] = 4.
        # The unscored value must not affect either the observed gain or its CI.
        arrays['mean'][1, 0] = 1e6
        arrays['background_mean'][1, 0] = 0.
        np.savez_compressed(path, **arrays)
    result = export_campaign_figures(campaign)
    rows = _rows(result['tables']['paired_block_bootstrap'])
    assert len(rows) == 4*2*2*2  # models x block sizes x resampling schemes x metrics
    for row in rows:
        assert float(row['rmse_gain_mm']) == pytest.approx(3.)
        assert float(row['gain_ci95_low_mm']) == pytest.approx(3.)
        assert float(row['gain_ci95_high_mm']) == pytest.approx(3.)
        assert int(row['requested_replicates']) == int(row['valid_replicates']) == 2000
        assert int(row['valid_station_hours']) == 7
    details = json.loads(Path(result['bootstrap_details']).read_text(encoding='utf-8'))
    assert details['seed'] == 20261122
    assert details['spatial_group_width_km'] == 50
    assert details['metrics'] == ['station_macro_rmse', 'micro_rmse']


def test_bootstrap_separates_macro_micro_and_resamples_whole_spatial_groups():
    from gnss_field.generative.report import _paired_bootstrap
    truth = np.zeros((2, 2))
    mean = np.array([[1., 3.], [1., 3.]])
    background = np.full((2, 2), 4.)
    mask = np.ones((2, 2), dtype=bool)
    xy = np.array([[0., 0.], [60000., 0.]])
    rows = _paired_bootstrap(truth, mean, background, mask, xy, np.array([1, 2]))
    assert rows == _paired_bootstrap(truth, mean, background, mask, xy, np.array([1, 2]))
    for row in rows:
        if row['metric']=='station_macro_rmse':
            assert row['rmse_gain_mm'] == pytest.approx(2.)
        else:
            assert row['rmse_gain_mm'] == pytest.approx(4.-np.sqrt(5.))
        if row['resampling']=='synchronous_time_blocks':
            assert row['gain_ci95_low_mm'] == pytest.approx(row['rmse_gain_mm'])
            assert row['gain_ci95_high_mm'] == pytest.approx(row['rmse_gain_mm'])
        else:
            # Drawing either of two singleton groups twice yields gain 3 or 1.
            assert row['gain_ci95_low_mm'] == pytest.approx(1.)
            assert row['gain_ci95_high_mm'] == pytest.approx(3.)
    with pytest.raises(ValueError, match='calendar'):
        _paired_bootstrap(truth, mean, background, mask, xy, np.array([1, 3]))
