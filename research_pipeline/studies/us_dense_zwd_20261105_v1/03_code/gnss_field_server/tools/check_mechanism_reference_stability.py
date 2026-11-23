"""Check candidate-vs-previous-product evidence under matched higher-K/noise."""
import time
import shutil
import torch
import validate_generator_mechanisms as experiment
import report_generator_mechanisms as reporting
from gnss_field.artifacts import read_json, write_json


def main():
    root, base = experiment.ROOT, experiment.base
    results = read_json(root/'confirmation_results.json')
    budget = read_json(root/'local_budget.json')
    base.PRIOR_SPENT = budget['consumed_seconds']
    base.CLOCK_START = time.monotonic()
    # Recover this helper's first Windows MAX_PATH failure without resetting cost.
    recovery = root/'reference_check/recovery.json'
    orphan = root/'reference_stability/gan/previous_joint_noise_k32.npz'
    if orphan.exists() and not recovery.exists():
        additional = max(0., orphan.stat().st_mtime-budget['updated'])
        base.PRIOR_SPENT += additional
        write_json(recovery, dict(reason='Atomic receipt temporary path263 characters exceeded Windows MAX_PATH',
                                 added_uncommitted_seconds=additional, remedy='Shorter filenames; trained weights unchanged'))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = base.PilotData()
    compared = {}
    for family, result in results['families'].items():
        if not result.get('probability_confirmed', False):
            continue
        previous = experiment.SOURCE/family/'joint_best.pt'
        comparisons = dict(primary=result['intervention_vs_previous_joint'])
        short = root/'reference_check'/family
        short.mkdir(parents=True, exist_ok=True)
        old = root/'reference_stability'/family/'previous_joint_k64'
        if old.with_suffix('.json').exists() and not (short/'high64.json').exists():
            for extension in ('.npz', '.json'):
                shutil.copy2(old.with_suffix(extension), (short/'high64').with_suffix(extension))
        high = experiment.evaluate_checkpoint(data, family, previous, short/'high64', 64)
        current = read_json((root/'endpoints64'/family/'intervention').with_suffix('.json'))
        comparisons['k64'] = reporting.screen(high, current)
        noise = experiment.evaluate_checkpoint(data, family, previous, short/'noise32', 32, 20261124)
        current = read_json((root/'noise_replication'/family/'intervention_k32').with_suffix('.json'))
        comparisons['independent_noise'] = reporting.screen(noise, current)
        compared[family] = dict(comparisons=comparisons,
                               previous_joint_advantage_stable=all(v['probability_screen'] for v in comparisons.values()))
    write_json(root/'reference_stability.json', dict(status='COMPLETE', families=compared))
    base.commit_budget('COMPLETE_WITH_REFERENCE_STABILITY')
    print('REFERENCE_STABILITY_COMPLETE', compared, flush=True)


if __name__ == '__main__':
    experiment.base.ROOT = experiment.ROOT
    with experiment.base.single_process():
        try:
            main()
        except Exception:
            if experiment.base.PRIOR_SPENT > 0:
                experiment.base.commit_budget('INCOMPLETE_REFERENCE_STABILITY')
            raise
