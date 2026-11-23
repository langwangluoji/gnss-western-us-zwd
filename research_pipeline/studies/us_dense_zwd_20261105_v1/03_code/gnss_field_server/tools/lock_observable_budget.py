"""Create a formal budget only from the new runner's real A800 benchmark."""
import argparse
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from gnss_field.artifacts import read_json, sha256, immutable_json
from gnss_field.training.release import verify_release
from gnss_field.training.observable import study_schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-arm-cap', type=float, default=.5)
    parser.add_argument('--evaluation-cap', type=float, default=.5)
    args = parser.parse_args()
    benchmark = read_json(args.benchmark)
    manifest_path = Path(args.bundle)/'manifest.json'
    manifest = read_json(manifest_path)
    project = Path(__file__).resolve().parents[1]
    runner = project/'src/gnss_field/training/observable.py'
    models = {arm: sha256(project/'src/gnss_field/models'/name)
              for arm, name in [('mean', 'mean_shift.py'), ('edm', 'subspace.py')]}
    if (not benchmark['formal_verification'] or not benchmark.get('real_100_plus100_equals200') or 'A800' not in benchmark['gpu'] or
        benchmark['bundle_manifest_sha256'] != sha256(manifest_path) or benchmark['updates'] != 20):
        raise ValueError('A real matching A800 benchmark is required; local RTX smoke cannot unlock paid training')
    if not benchmark.get('real_slurm_signal_resume'):
        raise ValueError('A real Slurm signal pause and resume probe is required')
    release_hash = verify_release(project)
    if benchmark.get('release_sha256') != release_hash:
        raise ValueError('Benchmark must match all published code, data and scoring dependencies')
    reference_arm = 'mean' if manifest.get('convergence_protocol') else 'edm'
    if (benchmark['arm'] != reference_arm or benchmark['runner_sha256'] != sha256(runner) or
        benchmark['model_sha256'] != models[reference_arm] or benchmark['model_sources_sha256'] != models):
        raise ValueError('The registered benchmark must match both deployed model sources and runner')
    updates, science = study_schedule(manifest, manifest.get('training_hours', 6288))
    evaluations = 1+(updates+science.evaluate_every-1)//science.evaluate_every
    estimate = (updates*benchmark['seconds_per_update']+evaluations*benchmark['development_seconds'])*1.5/3600
    if not all(math.isfinite(v) and v > 0 for v in (args.per_arm_cap, args.evaluation_cap)) or estimate > args.per_arm_cap:
        raise ValueError(f'Estimated {estimate:.4f} card h exceeds registered per-arm cap {args.per_arm_cap}')
    immutable_json(args.output, dict(A800_benchmark_verified=True,
        release_sha256=release_hash,
        benchmark_sha256=sha256(args.benchmark), bundle_manifest_sha256=sha256(manifest_path),
        runner_sha256=benchmark['runner_sha256'], maximum_updates_per_arm=updates,
        model_sha256=models, allowed_arms=manifest.get('allowed_arms', ['mean', 'edm']), upper_bound_reference_arm=reference_arm,
        registered_seed=42, registered_run_root=str(Path(args.output).resolve().parent),
        maximum_card_hours_per_arm=args.per_arm_cap, estimated_card_hours_per_arm_with_margin=estimate,
        budget_scope='formal training plus B development; engineering and C evaluation separately capped',
        stage_wall_caps_card_hours=dict(engineering=5/60, training_per_arm=args.per_arm_cap, evaluation_per_arm=args.evaluation_cap),
        maximum_default_mean_workflow_card_hours=5/60+args.per_arm_cap+args.evaluation_cap,
        maximum_concurrent_gpus=1, maximum_attempts=1,
        default_arm='mean', edm_coverage_counterfactual='optional separately registered job; no automatic GAN/search/long training'))


if __name__ == '__main__':
    main()
