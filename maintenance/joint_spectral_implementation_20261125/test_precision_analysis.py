import numpy as np

import precision_analysis as analysis


def test_direct_representation_contrast_uses_common_month_draws():
    records = [dict(case=c, seed=s, macro_rmse_mm=v) for c, s, v in
               [('A', 42, 9.7), ('A', 17, 9.9), ('S', 42, 9.9), ('S', 17, 10.)]]
    draws = {('A', 42): np.array([-.3, -.2, -.1]), ('A', 17): np.array([-.1, -.2, -.3]),
             ('S', 42): np.array([.1, .2, .3]), ('S', 17): np.array([.3, .2, .1])}
    result = analysis.matched_contrast(records, draws, 'A', 'S')
    assert abs(result['delta_macro_rmse_mm'] + .15) < 1e-12
    np.testing.assert_allclose(result['common_month_two_seed_paired95_mm'], [-.4, -.4])
    assert result['improves_both_seeds'] and result['negative_interval_upper']


def test_direct_representation_contrast_waits_for_both_seeds():
    assert analysis.matched_contrast([], {}, 'A', 'S') is None


def test_pooled_improvement_does_not_hide_opposing_seed_direction():
    records = [dict(case=c, seed=s, macro_rmse_mm=v) for c, s, v in
               [('A', 42, 9.5), ('A', 17, 10.1), ('S', 42, 10.), ('S', 17, 10.)]]
    draws = {(c, s): np.full(10, v-10.) for c, s, v in
             [('A', 42, 9.5), ('A', 17, 10.1), ('S', 42, 10.), ('S', 17, 10.)]}
    result = analysis.matched_contrast(records, draws, 'A', 'S')
    assert result['delta_macro_rmse_mm'] < 0
    assert not result['improves_both_seeds']


def test_candidate_gate_applies_increment_tolerance_to_each_seed():
    mean=dict(case='I',improves_both_seeds=True,negative_interval_upper=True)
    rows=[dict(case='I',seed=42,delta_increment_macro_rmse_mm=.011),
          dict(case='I',seed=17,delta_increment_macro_rmse_mm=-.02)]
    assert not analysis.candidate_gate(mean,rows)['eligible_for_next_validation']


def test_candidate_gate_requires_paired_point_support_and_both_seeds():
    mean=dict(case='I',improves_both_seeds=True,negative_interval_upper=True)
    rows=[dict(case='I',seed=42,delta_increment_macro_rmse_mm=0.)]
    assert not analysis.candidate_gate(mean,rows)['eligible_for_next_validation']
    rows.append(dict(case='I',seed=17,delta_increment_macro_rmse_mm=.005))
    assert analysis.candidate_gate(mean,rows)['eligible_for_next_validation']
    assert not analysis.candidate_gate(dict(mean,negative_interval_upper=False),rows)['eligible_for_next_validation']
