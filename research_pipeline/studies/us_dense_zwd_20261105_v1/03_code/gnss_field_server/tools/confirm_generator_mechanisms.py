"""Higher-K and adaptation-seed checks for the registered paired experiment."""
import json
import time
import numpy as np
import torch
import validate_generator_mechanisms as experiment
import report_generator_mechanisms as reporting
from gnss_field.artifacts import immutable_json, read_json, write_json, sha256


def main():
    root, base = experiment.ROOT, experiment.base
    if read_json(root/'paired_results.json')['status'] != 'COMPLETE':
        raise ValueError('Six arms are not complete')
    base.ROOT = root
    base.PRIOR_SPENT = read_json(root/'local_budget.json')['consumed_seconds']
    base.CLOCK_START = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = read_json(root/'experiment.lock.json')
    if sha256(experiment.__file__) != config['training_code'] or sha256(base.PROJECT/'src/gnss_field/objectives/ensemble_mean.py') != config['objective_code']:
        raise ValueError('Parent training or objective release changed')
    for family, expected in config['source_checkpoints'].items():
        if sha256(experiment.SOURCE/family/'last.pt') != expected:
            raise ValueError('Source checkpoint changed')
    if sha256(base.PROJECT/'src/gnss_field/models/conditioned_unet.py') != config['model_code']:
        raise ValueError('Source model changed')
    if sha256(base.DATA_ROOT/'data.npz') != config['input_cache_sha256']:
        raise ValueError('Input cache changed')
    immutable_json(root/'confirmation.lock.json', dict(parent=sha256(root/'experiment.lock.json'),
                   code=sha256(__file__), report_code=sha256(reporting.__file__), all_families_members=32,
                   candidate_members=64, extra_noise_seed=20261124, adaptation_seed=17,
                   scoring_rule='predeclared paired point/probability thresholds; station-macro CRPS primary, micro retained'))
    data = base.PilotData()
    for family in base.FACTORIES:
        for arm in ('source', 'control', 'intervention'):
            checkpoint = experiment.SOURCE/family/'last.pt' if arm == 'source' else root/'seed42'/family/arm/'last.pt'
            experiment.evaluate_checkpoint(data, family, checkpoint, root/'endpoints32'/family/arm, 32)
    results = reporting.report(32)
    confirmation = {}
    for family, result in results.items():
        if not (result['point_screen'] or result['probability_screen']):
            confirmation[family] = dict(status='NO_REGISTERED_GAIN_K32', primary=result)
            continue
        estimate = 2*config['updates']*read_json(root/'preflight.json')['timings'][family]['slower_seconds']+180
        if base.LOCAL_CAP-base.budget_seconds() < estimate:
            confirmation[family] = dict(status='INSUFFICIENT_REPLICATION_BUDGET', primary=result)
            continue
        write_json(root/'status.json', dict(stage='REPLICATION', family=family, updated=time.time()))
        for arm in experiment.ARMS:
            experiment.train_arm(data, family, arm, config, adaptation_seed=17)
        records = {}
        for arm in experiment.ARMS:
            records[f'seed17_{arm}'] = experiment.evaluate_checkpoint(data, family,
                       root/'seed17'/family/arm/'last.pt', root/'replication'/family/f'{arm}_k32', 32)
            records[f'noise_{arm}'] = experiment.evaluate_checkpoint(data, family,
                       root/'seed42'/family/arm/'last.pt', root/'noise_replication'/family/f'{arm}_k32', 32, 20261124)
            records[f'k64_{arm}'] = experiment.evaluate_checkpoint(data, family,
                       root/'seed42'/family/arm/'last.pt', root/'endpoints64'/family/arm, 64)
        replication = reporting.screen(records['seed17_control'], records['seed17_intervention'])
        noise = reporting.screen(records['noise_control'], records['noise_intervention'])
        high_k = reporting.screen(records['k64_control'], records['k64_intervention'])
        point_pass = reporting.confirmed_point_direction(result, high_k, replication, noise)
        probability_pass = result['probability_screen'] and high_k['probability_screen'] and replication['probability_screen'] and noise['probability_screen']
        reference = experiment.evaluate_checkpoint(data, family, experiment.SOURCE/family/'point_best.pt', root/'references'/family/'previous_point_k32', 32)
        joint_reference = experiment.evaluate_checkpoint(data, family, experiment.SOURCE/family/'joint_best.pt', root/'references'/family/'previous_joint_k32', 32)
        confirmation[family] = dict(status='SHORT_ADAPTATION_DIRECTION_CONFIRMED' if point_pass or probability_pass else 'DIRECTION_NOT_STABLE',
                  point_confirmed=bool(point_pass), probability_confirmed=bool(probability_pass),
                  primary=result, adaptation_seed17=replication, independent_noise=noise, high_k64=high_k,
                  previous_point_checkpoint=reference, previous_joint_checkpoint=joint_reference,
                  intervention_vs_previous_joint=reporting.screen(joint_reference, result['records']['intervention']),
                  intervention_vs_previous_point_rmse=result['records']['intervention']['scores']['withheld_weather_withheld']['macro_rmse']-reference['scores']['withheld_weather_withheld']['macro_rmse'])
        write_json(root/'confirmation_partial.json', confirmation)
    write_json(root/'confirmation_results.json', dict(status='COMPLETE', families=confirmation,
               interpretation='Local paired adaptation evidence; not independent full-chain seeds or server-scale convergence'))
    write_json(root/'status.json', dict(stage='COMPLETE_WITH_CONFIRMATION', updated=time.time(), conclusions={f: r['status'] for f, r in confirmation.items()}))
    base.commit_budget('COMPLETE_WITH_CONFIRMATION')
    print('MECHANISM_CONFIRMATION_COMPLETE', json.dumps({f: r['status'] for f, r in confirmation.items()}), flush=True)


if __name__ == '__main__':
    experiment.base.ROOT = experiment.ROOT
    with experiment.base.single_process():
        try:
            main()
        except Exception as error:
            write_json(experiment.ROOT/'confirmation_status.json', dict(stage='INCOMPLETE', error=str(error), updated=time.time()))
            if experiment.base.PRIOR_SPENT > 0:
                experiment.base.commit_budget('INCOMPLETE_CONFIRMATION')
            raise
