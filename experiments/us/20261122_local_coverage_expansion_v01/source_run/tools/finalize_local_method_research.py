"""Final evidence manifest; this is a development candidate, not a D2020 lock."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from local_probe_diagnostics import PROJECT, MEAN
from gnss_field.artifacts import immutable_json, read_json, sha256, write_json


def main():
    root = PROJECT/'results/20261119_local_method_research_v02'
    fused = PROJECT/'results/20261119_local_fused_probability_v02'
    student = PROJECT/'results/20261119_local_student_probability_v01'
    mean_hash = sha256(MEAN)
    assert mean_hash == '605650e000ce4a0ab31d3d85996fd16316c978547fe9f2821c65c346cd1b86f0'
    blend = read_json(root/'full2019_blend/results.json')
    probability = read_json(fused/'full2019/results.json')
    parameters = read_json(fused/'B_only_parameters.lock.json')
    assert probability['n_hours'] == 8760 and probability['n_station_hours'] == 642258
    assert probability['joint']['correlated']['n_groups'] == 6
    assert probability['joint']['correlated']['n_group_hours'] == 49908
    assert abs(probability['analytic_common_marginal']['primary']['station_macro_rmse_mm']-blend['metrics']['blend']['station_macro_rmse_mm']) < 1e-8
    result_files = sorted((fused/'full2019').glob('forecast_2019-*.npz'))
    all_hours, members = [], 0
    for path in result_files:
        z = dict(np.load(path))
        assert z['samples'].shape[1] == 128
        assert np.isfinite(z['samples'].transpose(0, 2, 1)[z['population']]).all()
        all_hours.append(z['hours'])
        members += z['samples'].shape[0]*z['samples'].shape[1]
    hours = np.concatenate(all_hours)
    np.testing.assert_array_equal(hours, np.arange(hours[0], hours[0]+8760))
    tests = ET.parse(root/'verification_tests_v02.xml').getroot()
    suites = list(tests.iter('testsuite'))
    assert sum(int(v.get('failures', 0))+int(v.get('errors', 0)) for v in suites) == 0
    assert sum(int(v.get('tests', 0)) for v in suites) == 60
    conditional = PROJECT/'results/20261119_local_conditional_probability_v01'
    conditional_result = read_json(conditional/'results.json')
    assert conditional_result['marginal']['primary']['n_station_hours'] == 642258
    for baseline in result_files:
        a, b = dict(np.load(baseline)), dict(np.load(conditional/baseline.name))
        np.testing.assert_array_equal(a['hours'], b['hours'])
        np.testing.assert_array_equal(a['station_ids'], b['station_ids'])
        np.testing.assert_array_equal(a['population'], b['population'])
        np.testing.assert_array_equal(a['analytic_mean'], b['analytic_mean'])
        assert np.isfinite(b['samples'].transpose(0, 2, 1)[b['population']]).all()
        assert (b['std'][b['population']] > 0).all()
    snapshots = root/'code_snapshot'
    snapshots.mkdir(exist_ok=True)
    names = ['local_shared_field.py', 'local_method_nested.py', 'local_joint_field.py',
             'local_probability_experiment.py', 'local_probability_full2019.py', 'local_probability_aggregate.py',
             'local_student_field.py', 'local_observable_subspace.py', 'report_local_method_research.py',
             'local_conditional_gaussian.py', 'local_condition_diagnostics.py', 'report_conditional_field.py',
             'finalize_local_method_research.py']
    for name in names:
        (snapshots/name).write_bytes((PROJECT/'tools'/name).read_bytes())
    write_json(student/'input_verification.json', dict(post_fit_verification=True,
        interpretation='Supplement to original preregistration; no claim these hashes were recorded before fitting',
        gaussian_aggregation_lock_sha256=sha256(fused/'full2019/aggregation.lock.json'),
        gaussian_monthly_inputs_sha256={p.name: sha256(p) for p in result_files},
        student_parameters_sha256=sha256(student/'B_only_parameters.lock.json'),
        student_result_sha256=sha256(student/'results.json'), full_year_hours=8760,
        common_population_verified=True))
    immutable_json(root/'selected_method.lock.json', dict(status='local_development_candidate',
        mean_checkpoint_sha256=mean_hash, mean_formula='(1-w)*M0+w*A_IDW on Z0 grid', w=parameters['weight'],
        covariance_parameters=parameters['model'], covariance_parameter_hash=sha256(fused/'B_only_parameters.lock.json'),
        spatial_basis=dict(kind='fixed compact bisquare', spacings_km=[100, 300], radii_km=[150, 450], ranks=[1049, 165]),
        input='A174, joint five-variable mask, existing M0 four coarse frames, fixed geometry and terrain',
        observation_noise='independent surface unresolved plus observation error, excluded from grid members',
        latent_law='Gaussian', members=128, seed=42, retrospective_evaluation_authorized=False,
        future_generator=dict(trained=False, hypothesis='R64 observation-supported modes plus Gaussian remainder',
                              EDM_trained=False, GAN_trained=False)))
    immutable_json(root/'selected_method_v02.lock.json', dict(status='post_diagnostic_local_development_candidate',
        point_baseline_sha256=sha256(root/'selected_method.lock.json'), mean_formula='(1-w)*M0+w*A_IDW',
        probability='A-conditioned amplitude, locally correlated Gaussian latent field plus declared unresolved observation error',
        conditional_scale_parameters=read_json(conditional/'B_only_parameters.lock.json'),
        conditional_parameters_sha256=sha256(conditional/'B_only_parameters.lock.json'),
        conditional_protocol_sha256=sha256(conditional/'protocol.lock.json'),
        conditional_result_sha256=sha256(conditional/'results.json'),
        independent_review_sha256=sha256(conditional/'review_crossed.json'),
        future_generator='R64 fixed-prior observation-sensitive modes plus Gaussian remainder; not trained',
        members=128, seed=42, retrospective_evaluation_authorized=False,
        selection_note='C2019 motivated and evaluated scale family; parameters fitted only on B2018'))
    write_json(root/'verification.json', dict(verified=True, tests_passed=60, M0_unchanged=True,
        station_hours=642258, annual_hours=8760, stored_fused_station_members=members,
        code_sha256={name: sha256(snapshots/name) for name in names},
        selected_method_sha256=sha256(root/'selected_method_v02.lock.json'),
        report_sha256=sha256(root/'RESULTS.md'), D2020_accessed=False,
        interpretation='Local numerical and development-evaluation verification; no final generalization claim'))
    print('Verified:60 tests; frozen M0;8760h;642258 point station-hours;conditional Gaussian development candidate;D2020 sealed', flush=True)


if __name__ == '__main__':
    main()
