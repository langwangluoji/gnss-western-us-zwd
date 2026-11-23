"""Audited resource addendum; the published training release remains unchanged."""
import argparse
import math
from pathlib import Path
import torch
from gnss_field.artifacts import read_json, immutable_json, sha256
from gnss_field.training.release import verify_release
from gnss_field.training.observable import _same


def measured_estimate(benchmark, timing):
    if len(timing) != 2 or any(record['step'] != 200 for record in timing):
        raise ValueError('Both actual warmed 200-update checkpoints are required')
    values = [record['elapsed'] for record in timing]+[benchmark[key] for key in
        ('seconds_per_update', 'development_seconds', 'checkpoint_seconds')]
    if not all(math.isfinite(v) and v >= 0 for v in values) or min(values[:2]) <= 0 or benchmark['updates'] != 20:
        raise ValueError('Invalid measured timing')
    rate = max(record['elapsed']/record['step'] for record in timing)
    evaluations = 51  # zero, 100,...,5000; only EMA can select.
    cold = benchmark['seconds_per_update']*benchmark['updates']
    components = dict(cold_probe_once=cold, warmed_updates=5000*rate,
        B_development=evaluations*benchmark['development_seconds'],
        checkpoint_io=(3*evaluations+52)*benchmark['checkpoint_seconds'],
        allocation_loading_allowance=120.)
    return dict(steady_seconds_per_update=rate, components_seconds=components,
        estimated_seconds_with_margin=1.5*sum(components.values()), margin=1.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    release, run = Path(args.release).resolve(), Path(args.run).resolve()
    registration = read_json(run/'repair_budget_v01/registration.json')
    if sha256(__file__) != registration['estimator_sha256'] or verify_release(release) != registration['release_sha256']:
        raise ValueError('Resource addendum or frozen training release differs')
    for name, digest in registration['evidence_sha256'].items():
        if sha256(run/name) != digest:
            raise ValueError('Original real A800 timing evidence differs')
    benchmark = read_json(run/'benchmark/benchmark.json')
    manifest = read_json(release/'bundle/manifest.json')
    runner = release/'src/gnss_field/training/observable.py'
    model_hashes = {arm: sha256(release/'src/gnss_field/models'/name) for arm, name in
        [('mean', 'mean_shift.py'), ('edm', 'subspace.py')]}
    if (not benchmark['formal_verification'] or benchmark['platform'] != 'Linux' or 'A800' not in benchmark['gpu']
        or not benchmark['real_slurm_signal_resume'] or not benchmark['real_100_plus100_equals200']
        or benchmark['arm'] != 'mean' or benchmark['release_sha256'] != registration['release_sha256']
        or benchmark['runner_sha256'] != sha256(runner) or benchmark['model_sources_sha256'] != model_hashes
        or benchmark['model_sha256'] != model_hashes['mean']
        or benchmark['bundle_manifest_sha256'] != sha256(release/'bundle/manifest.json')
        or manifest.get('convergence_protocol') != 'mean_convergence_v1' or manifest['maximum_updates_per_arm'] != 5000
        or manifest['allowed_arms'] != ['mean']):
        raise ValueError('Real A800/model/data/scientific checks are required')
    states = [torch.load(run/f'benchmark/{name}/last.pt', map_location='cpu', weights_only=False)
        for name in ('resume_continuous', 'resume_split')]
    for state in states:
        if (state['spec']['stage'] != 'observable_mean_benchmark' or state['spec']['seed'] != 42
            or state['spec']['dependencies']['release_sha256'] != registration['release_sha256']
            or state['spec']['dependencies']['bundle_manifest'] != sha256(release/'bundle/manifest.json')):
            raise ValueError('Warmed engineering timing contract differs')
    for key in ('model', 'ema', 'optimizers', 'rng', 'cursor', 'step'):
        _same(states[0][key], states[1][key])
    estimate = measured_estimate(benchmark, states)
    if estimate['estimated_seconds_with_margin'] > 900:
        raise ValueError('Corrected measured estimate still exceeds unchanged 15-minute cap')
    immutable_json(run/'budget.lock.json', dict(A800_benchmark_verified=True,
        release_sha256=registration['release_sha256'], benchmark_sha256=sha256(run/'benchmark/benchmark.json'),
        bundle_manifest_sha256=sha256(release/'bundle/manifest.json'), runner_sha256=sha256(runner),
        maximum_updates_per_arm=5000, model_sha256=model_hashes, allowed_arms=['mean'],
        upper_bound_reference_arm='mean', registered_seed=42, registered_run_root=str(run),
        maximum_card_hours_per_arm=.25, estimated_card_hours_per_arm_with_margin=estimate['estimated_seconds_with_margin']/3600,
        budget_scope='Formal training plus B development; original real A800 evidence; cold initialization charged once',
        stage_wall_caps_card_hours=dict(engineering=5/60, training_per_arm=.25, evaluation_per_arm=.25),
        maximum_default_mean_workflow_card_hours=35/60, maximum_concurrent_gpus=1, maximum_attempts=1,
        default_arm='mean', automatic_EDM=False, automatic_GAN=False,
        resource_addendum_sha256=sha256(run/'repair_budget_v01/registration.json'),
        estimator_sha256=sha256(__file__), timing_evidence_sha256=registration['evidence_sha256'], measurement=estimate))
    print('BUDGET_REPAIR_VERIFIED', estimate, flush=True)


if __name__ == '__main__':
    main()
