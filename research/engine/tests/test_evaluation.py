import numpy as np
import pandas as pd
import pytest

from gnss_field.evaluation import (
    point_metrics, evaluate_arrays, crps_ensemble, energy_score_ensemble,
    variogram_score, paired_block_bootstrap, select_products, direction_gate,
    sampling_stability, build_product_lock, validate_retrospective_access,
    ssim_diagnostic, psd_diagnostic, holm_adjust, crossed_increment_bootstrap,
    event_block_bootstrap, probability_gate, select_bstar, selection_values,
)


def metadata(n):
    return pd.DataFrame(dict(x=np.arange(n)*20000., y=np.zeros(n),
                             height=np.zeros(n), dense_group=['g']*n, cohort=['C']*n))


def test_station_macro_differs_from_micro_and_invalid_predictions_raise():
    y = np.zeros((3, 2))
    p = np.array([[1., 3.], [1., np.nan], [1., np.nan]])
    valid = np.isfinite(p)
    result = point_metrics(y, p, valid)
    assert result['station_macro_rmse_mm'] == pytest.approx(2)
    assert result['micro_rmse_mm'] == pytest.approx(np.sqrt(3))
    assert result['n_station_hours'] == 4
    p[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        point_metrics(y, p, valid)


def test_finite_ensemble_crps_and_joint_dependence():
    assert crps_ensemble(0., np.array([0., 2.])) == pytest.approx(.5)
    y = np.zeros(4)
    coherent = np.array([[-1., -1., -1., -1.], [1., 1., 1., 1.]])
    shuffled = np.array([[-1., 1., -1., 1.], [1., -1., 1., -1.]])
    observed = np.ones(4)
    assert np.allclose(crps_ensemble(observed, coherent, axis=0),
                       crps_ensemble(observed, shuffled, axis=0))
    assert energy_score_ensemble(observed, shuffled) > energy_score_ensemble(observed, coherent)
    d = np.abs(np.arange(4)[:, None]-np.arange(4)[None, :])*20.
    assert variogram_score(observed, shuffled, d) > variogram_score(observed, coherent, d)
    assert energy_score_ensemble(y, np.zeros((2, 4))) == 0


def test_scope_intervals_fine_common_population_and_no_fake_gaussian_joint():
    y = np.zeros((2, 4))
    f = np.ones_like(y, bool)
    s0 = f.copy()
    s0[1, 3] = False
    samples = np.broadcast_to(np.array([0., 10.])[None, :, None], (2, 2, 4)).copy()
    out = evaluate_arrays(np.arange(2), np.arange(4), y, y, f, s0,
                          samples=samples, metadata=metadata(4),
                          coarsened_prediction=y+1., safe_T=np.array([1, 1, 1, 0], bool))
    assert out['full']['n_station_hours'] == 8
    assert out['primary']['n_station_hours'] == 7
    assert out['unsupported']['n_station_hours'] == 1
    assert out['fine']['model']['n_station_hours'] == 6
    assert out['fine']['coarsened']['n_station_hours'] == 6
    assert out['full']['intervals']['90']['mean_width_mm'] == pytest.approx(9.)
    assert out['full']['intervals']['90']['coverage'] == 0
    assert out['joint']['n_group_hours'] == 1
    gaussian = evaluate_arrays(np.arange(2), np.arange(4), y, y, f, s0,
                               std=np.ones_like(y), metadata=metadata(4))
    assert 'joint' not in gaussian
    assert gaussian['full']['intervals']['90']['coverage'] == 1


def test_prediction_failure_cannot_shrink_full_population():
    y = np.zeros((1, 4))
    p = y.copy()
    p[0, 3] = np.nan
    with pytest.raises(ValueError, match='finite'):
        evaluate_arrays([0], np.arange(4), y, p, np.ones_like(y, bool),
                        np.array([[1, 1, 1, 0]], bool), metadata=metadata(4))


def test_pairs_include_across_distance_bins_and_revert_height():
    y = np.zeros((1, 4))
    p = np.array([[0., 1., 2., 3.]])
    m = metadata(4)
    m['height'] = 3000.
    out = evaluate_arrays([0], np.arange(4), y, p, np.ones_like(y, bool),
                          np.ones_like(y, bool), metadata=m)
    assert out['increments']['0-25']['n_pair_hours'] == 3
    assert out['increments']['25-50']['n_pair_hours'] == 2
    assert out['increments']['0-50']['n_pair_hours'] == 5
    assert out['increments']['0-50']['group_macro_rmse_mm'] == pytest.approx(np.e*np.sqrt(11/5))


def test_paired_bootstrap_synchronizes_stations_and_recomputes_rmse():
    y = np.zeros((4, 2))
    p = np.array([[0., 0.], [0., 0.], [4., 4.], [4., 4.]])
    b = np.ones_like(y)*2.
    result = paired_block_bootstrap(np.arange(4)*24, y, p, b, np.ones_like(y, bool),
                                    replicates=20, block_days=2, return_indices=True)
    assert result['difference_mm'] == pytest.approx(np.sqrt(8)-2)
    for indices, value in zip(result['indices'], result['replicate_differences_mm']):
        expected = np.sqrt(np.mean(p[indices, 0]**2))-2
        assert value == pytest.approx(expected)
    assert set(np.round(result['replicate_differences_mm'], 8)) <= {-2., round(np.sqrt(8)-2, 8), 2.}


def test_dual_products_exclude_pretraining_and_startup_and_keep_unrounded_order():
    rows = [dict(checkpoint='pre', stage='pretrain', step=9999, station_macro_rmse_mm=0., group_macro_es_mm=0.),
            dict(checkpoint='startup', stage='refine', step=200, station_macro_rmse_mm=0., group_macro_es_mm=0.),
            dict(checkpoint='point', stage='refine', step=1000, station_macro_rmse_mm=1.00001, group_macro_es_mm=2.),
            dict(checkpoint='joint', stage='refine', step=2000, station_macro_rmse_mm=1.00002, group_macro_es_mm=1.)]
    result = select_products(rows, products=('point', 'joint'))
    assert result['point']['selected']['checkpoint'] == 'point'
    assert result['joint']['selected']['checkpoint'] == 'joint'
    assert len(result['point']['top_candidates']) == 2


def test_stability_prefix_and_direction_gates_and_holm():
    a = np.arange(256.).reshape(1, 128, 2)
    stable = sampling_stability(a[:, :64], a, point64_mm=1., point128_mm=1.001,
                                order64=['a'], order128=['a'], gates64={'x': True}, gates128={'x': True},
                                member_ids64=np.arange(64), member_ids128=np.arange(128),
                                noise_provenance64={'seed': 42}, noise_provenance128={'seed': 42})
    assert stable['stable']
    assert not sampling_stability(a[:, :64], a+1, point64_mm=1., point128_mm=1.,
                                   order64=[], order128=[], gates64={}, gates128={})['prefix_matches']
    comparisons = {str(i): {'difference_mm': -.1, 'ci95_mm': [-.3, -.01]} for i in range(4)}
    assert direction_gate(comparisons)['fine_pass']
    comparisons['0']['ci95_mm'][1] = 0
    assert not direction_gate(comparisons)['fine_pass']
    assert direction_gate(comparisons, formal=False)['fine_pass']
    assert holm_adjust([.01, .04, .03]) == pytest.approx([.03, .06, .06])


def test_product_lock_enforces_development_and_unchanged_parameters():
    records = [dict(checkpoint='a', checkpoint_sha256='b'*64, stage='refine', step=1000,
                    year=2019, full_year=True, station_macro_rmse_mm=1.)]
    products = select_products(records, products=('point',), require_full_year=True)
    lock = build_product_lock(products, config_sha256='a'*64, manifest_sha256='c'*64,
                              stability={'stable': True, 'members': 64})
    validate_retrospective_access(lock, year=2020, role='D', config_sha256='a'*64,
                                  manifest_sha256='c'*64, checkpoint_sha256='b'*64)
    with pytest.raises(ValueError):
        validate_retrospective_access(lock, year=2020, role='D', config_sha256='d'*64,
                                      manifest_sha256='c'*64, checkpoint_sha256='b'*64)


def test_fixed_range_ssim_and_phase_sensitive_psd():
    axis = np.arange(32)
    a = np.cos(2*np.pi*axis[None, :]/4)*np.ones((32, 1))
    b = -a
    assert ssim_diagnostic(a, a, np.ones_like(a, bool), data_range=2.)['ssim'] == pytest.approx(1)
    out = psd_diagnostic(a, b)
    assert out['phase_rmse'] > 1
    assert out['prediction_power'] == pytest.approx(out['truth_power'])
    assert out['squared_error_power'] > 0


def test_crossed_group_and_whole_event_bootstrap_are_paired():
    truth_increment = np.zeros((4, 2))
    pred_increment = np.array([[1., 3.]]*4)
    baseline = pred_increment*2
    out = crossed_increment_bootstrap(np.arange(4)*24, truth_increment, pred_increment,
                                      baseline, np.ones((4, 2), bool), ['a', 'b'], replicates=20)
    assert out['difference_mm'] == pytest.approx(-2.)
    assert np.max(out['replicate_differences_mm']) < 0
    event = event_block_bootstrap([1., 2., 3., 4.], [2., 3., 4., 5.],
                                  ['storm1', 'storm1', 'storm2', 'storm2'], replicates=20)
    assert event['difference'] == -1.
    assert event['ci95'] == [-1., -1.]


def test_gaussian_scores_are_mm_and_empty_scopes_remain_explicit():
    y = np.zeros((1, 4))
    f = np.ones_like(y, bool)
    out = evaluate_arrays([0], np.arange(4), y, y, f, f, std=np.ones_like(y), metadata=metadata(4))
    assert out['primary']['station_macro_crps_mm'] == pytest.approx(.23369497725510913)
    assert out['primary']['station_macro_gaussian_nll'] == pytest.approx(.9189385332046727)
    assert out['unsupported']['station_macro_rmse_mm'] is None
    assert out['unsupported']['intervals']['90']['coverage'] is None
    assert selection_values(out)['marginal']['value'] == pytest.approx(.23369497725510913)
    assert 'joint' not in selection_values(out)


def test_nan_outside_F_allowed_but_member_failure_inside_F_raises():
    y = np.array([[0., 0., 0., np.nan]])
    f = np.array([[1, 1, 1, 0]], bool)
    members = np.zeros((1, 2, 4))
    members[:, :, 3] = np.nan
    out = evaluate_arrays([0], np.arange(4), y, y, f, f, samples=members, metadata=metadata(4))
    assert out['full']['station_macro_crps_mm'] == 0.
    members[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        evaluate_arrays([0], np.arange(4), y, y, f, f, samples=members, metadata=metadata(4))


def test_probability_gate_requires_all_levels_and_formal_coverage_intervals():
    intervals = {str(p): {'coverage': p/100} for p in (80, 90, 95)}
    result = probability_gate(es_difference=-1., vs_difference=-1., crps_difference=0.,
                              intervals=intervals, formal=True)
    assert result['joint_improves']
    assert not result['calibrated_joint_pass']
    assert result['formal_coverage_status'] == 'indeterminate'
    ci = {str(p): [-.02, .02] for p in (80, 90, 95)}
    assert probability_gate(es_difference=-1., vs_difference=-1., crps_difference=0.,
                            intervals=intervals, formal=True, coverage_difference_ci=ci)['calibrated_joint_pass']


def test_bstar_full_year_registration_and_unrounded_step_tie():
    rows = [dict(model='M0', checkpoint='later', stage='refine', step=2000,
                 year=2019, full_year=True, station_macro_rmse_mm=1.),
            dict(model='M0', checkpoint='earlier', stage='refine', step=1000,
                 year=2019, full_year=True, station_macro_rmse_mm=1.),
            dict(model='unregistered', checkpoint='fast', stage='refine', step=1000,
                 year=2019, full_year=True, station_macro_rmse_mm=.1)]
    assert select_bstar(rows, ['M0'])['checkpoint'] == 'earlier'
    rows[1]['full_year'] = False
    assert select_bstar(rows, ['M0'])['checkpoint'] == 'later'


def test_bstar_accepts_registered_nontrained_point_controls():
    controls = [dict(model='A_IDW', checkpoint='fixed-parameters', step=0,
                     year=2019, full_year=True, station_macro_rmse_mm=.5),
                dict(model='M0', checkpoint='model', stage='refine', step=1000,
                     year=2019, full_year=True, station_macro_rmse_mm=1.)]
    assert select_bstar(controls, ['A_IDW', 'M0'])['model'] == 'A_IDW'


def test_covariance_failure_gaussian_std_and_psd_holes_are_explicit():
    y = np.zeros((1, 4))
    f = np.ones_like(y, bool)
    with pytest.raises(ValueError, match='positive'):
        evaluate_arrays([0], np.arange(4), y, y, f, f, std=y, metadata=metadata(4))
    image = np.zeros((8, 8))
    mask = np.ones_like(image, bool)
    mask[3, 3] = False
    with pytest.raises(ValueError, match='rectangle'):
        psd_diagnostic(image, image, mask)


def test_sampling_stability_requires_matching_declared_global_member_ids():
    a = np.zeros((1, 128, 2))
    with pytest.raises(ValueError, match='member'):
        sampling_stability(a[:, :64], a, point64_mm=1., point128_mm=1.,
                            order64=[], order128=[], gates64={}, gates128={},
                            member_ids64=np.arange(64), member_ids128=np.arange(128)[::-1])


def test_point_bootstrap_rejects_missing_original_stations_and_records_counts():
    population = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], bool)
    y = np.zeros((4, 2))
    prediction = np.broadcast_to([1., 10.], y.shape)
    result = paired_block_bootstrap(np.arange(4)*24, y, prediction, y, population,
                                    block_days=2, replicates=30, return_indices=True)
    assert result['difference_mm'] == 5.5
    assert np.all(result['replicate_differences_mm'] == 5.5)
    assert result['n_stations'] == 2
    assert result['rejected_draws'] > 0
    assert result['attempts'] == 30+result['rejected_draws']
    assert result['rejection_rate'] == result['rejected_draws']/result['attempts']
    for indices in result['indices']:
        assert population[indices].any(axis=0).all()


def test_point_bootstrap_exhaustion_reports_bounded_attempts():
    population = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], bool)
    y = np.zeros((4, 2))
    with pytest.raises(RuntimeError, match='attempt') as caught:
        paired_block_bootstrap(np.arange(4)*24, y, y+1., y, population,
                               block_days=2, replicates=1, seed=0, max_attempts=1)
    assert caught.value.attempts == 1
    assert caught.value.rejected_draws == 1
    assert caught.value.accepted_replicates == 0


def test_crossed_bootstrap_fixes_eligible_endpoint_groups_before_sampling():
    y = np.zeros((4, 2))
    population = np.broadcast_to([True, False], y.shape)
    out = crossed_increment_bootstrap(np.arange(4)*24, y, y+2., y, population,
                                      ['a', 'b'], replicates=30)
    assert out['n_groups'] == 1
    assert out['eligible_group_ids'] == ['a']
    assert out['excluded_group_ids'] == ['b']
    assert out['ci95_mm'] == [2., 2.]


def test_crossed_bootstrap_rejects_time_draws_missing_selected_group_weights():
    y = np.zeros((4, 2))
    population = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], bool)
    out = crossed_increment_bootstrap(np.arange(4)*24, y, y+3., y, population,
                                      ['a', 'b'], block_days=2, replicates=50)
    assert np.all(out['replicate_differences_mm'] == 3.)
    assert out['rejected_draws'] > 0
    assert out['attempts'] == 50+out['rejected_draws']
    assert out['n_groups'] == 2


def test_crossed_bootstrap_exhaustion_is_bounded_and_reports_counts():
    y = np.zeros((4, 2))
    population = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], bool)
    with pytest.raises(RuntimeError, match='attempt') as caught:
        crossed_increment_bootstrap(np.arange(4)*24, y, y+1., y, population,
                                    ['a', 'b'], block_days=2, replicates=1, seed=0, max_attempts=1)
    assert caught.value.attempts == 1
    assert caught.value.rejected_draws == 1


@pytest.mark.parametrize('step', [0, 1, 100, 199, 200])
def test_bstar_learned_startup_exclusion_preserves_fixed_controls(step):
    common = dict(model='M0', stage='refine', year=2019, full_year=True)
    rows = [dict(common, checkpoint='startup', step=step, station_macro_rmse_mm=.1),
            dict(common, checkpoint='eligible', step=1000, station_macro_rmse_mm=1.)]
    assert select_bstar(rows, ['M0'])['checkpoint'] == 'eligible'


def test_lock_normalizes_uppercase_selected_checkpoint_hash():
    records = [dict(checkpoint='a', checkpoint_sha256='B'*64, stage='refine', step=1000,
                    year=2019, full_year=True, station_macro_rmse_mm=1.)]
    products = select_products(records, products=('point',), require_full_year=True)
    lock = build_product_lock(products, config_sha256='A'*64, manifest_sha256='C'*64,
                              stability={'stable': True, 'members': 64})
    assert lock['products']['point']['checkpoint_sha256'] == 'b'*64
    assert products['point']['selected']['checkpoint_sha256'] == 'B'*64
    assert validate_retrospective_access(lock, year=2020, role='D', config_sha256='A'*64,
                                         manifest_sha256='C'*64, checkpoint_sha256='B'*64)


def test_sampling_prefix_accepts_fp32_roundoff_with_fixed_noise_provenance():
    a = np.full((1, 128, 2), 53.5, dtype=np.float32)
    b = a.copy()
    b[:, :64] += np.float32(.0005)
    result = sampling_stability(a[:, :64], b, point64_mm=1., point128_mm=1.001,
                                order64=['a'], order128=['a'], gates64={}, gates128={},
                                member_ids64=np.arange(64), member_ids128=np.arange(128),
                                noise_provenance64={'seed': 42, 'sampler': 'fixed-member-v1'},
                                noise_provenance128={'seed': 42, 'sampler': 'fixed-member-v1'})
    assert result['stable']
    assert 0 < result['prefix_max_abs_difference_mm'] <= .001
    assert result['prefix_tolerance_mm'] == .001
    assert result['tolerance_mm'] == .005


def test_sampling_prefix_rejects_unrelated_constant_outputs_and_missing_provenance():
    a = np.full((1, 128, 2), 53.5, dtype=np.float32)
    common = dict(point64_mm=1., point128_mm=1., order64=[], order128=[], gates64={}, gates128={})
    missing = sampling_stability(a[:, :64], a, **common)
    assert not missing['stable']
    assert not missing['provenance_matches']
    unrelated = sampling_stability(a[:, :64], a, **common,
                                   member_ids64=np.arange(64), member_ids128=np.arange(128),
                                   noise_provenance64={'seed': 42}, noise_provenance128={'seed': 43})
    assert not unrelated['stable']
    assert not unrelated['provenance_matches']


def test_sampling_prefix_tolerance_is_independent_of_rmse_tolerance():
    a = np.full((1, 128, 2), 53.5)
    b = a.copy()
    b[:, :64] += .0011
    result = sampling_stability(a[:, :64], b, point64_mm=1., point128_mm=1.,
                                order64=[], order128=[], gates64={}, gates128={},
                                member_ids64=np.arange(64), member_ids128=np.arange(128),
                                noise_provenance64={'seed': 42}, noise_provenance128={'seed': 42})
    assert not result['stable']
    assert not result['prefix_matches']


def test_sampling_prefix_tolerance_cannot_exceed_registered_roundoff_bound():
    a = np.zeros((1, 128, 2))
    with pytest.raises(ValueError, match='0.001'):
        sampling_stability(a[:, :64], a, point64_mm=1., point128_mm=1.,
                            order64=[], order128=[], gates64={}, gates128={}, prefix_tolerance_mm=.002)


def test_point_bootstrap_wholly_empty_draw_uses_rejection_counter():
    y = np.zeros((2, 1))
    with pytest.raises(RuntimeError) as caught:
        paired_block_bootstrap([0, 24], y, y+1., y, np.array([[True], [False]]),
                               replicates=1, block_days=1, seed=0, max_attempts=1)
    assert caught.value.rejected_draws == 1
    assert caught.value.accepted_replicates == 0
