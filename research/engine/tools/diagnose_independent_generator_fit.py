"""Post-training fixed-panel diagnostics; never select a new checkpoint here."""
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_independent_generators as train
from gnss_field.artifacts import immutable_json, write_json, read_json, sha256


def main():
    root = train.ROOT
    if read_json(root/'generators_results.json')['status'] != 'COMPLETE':
        raise RuntimeError('Wait for the complete three-model short-training queue')
    train.PRIOR_SPENT = read_json(root/'local_budget.json')['consumed_seconds']
    train.CLOCK_START = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if sha256(train.DATA_ROOT/'data.npz') != read_json(train.DATA_ROOT/'data_complete.json')['sha256']:
        raise ValueError('Input data changed since training')
    data = train.PilotData()
    fit, held = data.a['split_fit'], data.a['split_inner']
    allowed = train.qualified_training_rows(data.a['mask'], data.rows('train'), fit)
    panels = dict(training=allowed[np.rint(np.linspace(0, len(allowed)-1, 24)).astype(int)],
                  withheld_weather=data.rows('inner').reshape(12, -1)[:, [0, 6]].ravel())
    ids = np.concatenate([fit, held])
    out = root/'fit_diagnosis'
    out.mkdir(exist_ok=True)
    contract = dict(source_contract=sha256(root/'generators.lock.json'), members=8, seed=20261123,
                    diagnosis_code=sha256(__file__),
                    panels={k: data.hours[v] for k, v in panels.items()},
                    fitting_stations=fit, withheld_stations=held,
                    purpose='Post hoc fixed-panel optimization and fit diagnosis; no checkpoint selection; C/D not scored')
    immutable_json(out/'diagnosis.lock.json', contract)
    results = {}
    for family in train.FACTORIES:
        model = train.make_model(family)
        family_results = {}
        for state_name in ('initial', 'point_best', 'joint_best', 'last'):
            step, digest = 0, contract['source_contract']
            if state_name != 'initial':
                path = root/family/f'{state_name}.pt'
                state = torch.load(path, map_location='cuda', weights_only=False)
                if state['contract_sha256'] != contract['source_contract'] or state['family'] != family:
                    raise ValueError('Checkpoint provenance mismatch')
                model.load_state_dict(state['model'])
                step, digest = state['step'], sha256(path)
            # Equal-step products contain the same weights in this training path.
            previous = next((v for v in family_results.values() if v['step'] == step), None)
            if previous is not None:
                family_results[state_name] = dict(previous, checkpoint_sha256=digest)
                continue
            record = dict(step=step, checkpoint_sha256=digest, panels={})
            for name, rows in panels.items():
                samples = train.sample_population(model, data, rows, ids, 8)
                truth = data.a['labels'][np.ix_(rows, ids)]
                mask = data.a['mask'][np.ix_(rows, ids)]
                for role, columns in [('fitting_stations', np.arange(len(fit))),
                                      ('withheld_stations', np.arange(len(fit), len(ids)))]:
                    value = train.distribution_metrics(samples[:, :, columns], truth[:, columns],
                                                       mask[:, columns], data.a['xy'][ids[columns]])
                    record['panels'][f'{name}_{role}'] = value
            curve = read_json(root/family/'curve.json')
            original = next(v['validation'] for v in curve if v['step'] == step)
            recomputed = record['panels']['withheld_weather_withheld_stations']
            for metric in ('macro_rmse', 'ES_mm', 'CRPS_mm', 'group_ES_mm'):
                if not np.isclose(original[metric], recomputed[metric], atol=.00001, rtol=0):
                    raise ValueError(f'Archived internal metric was not reproduced: {family}/{step}/{metric}')
            record['archived_internal_scores_reproduced'] = True
            family_results[state_name] = record
            write_json(out/f'{family}.json', family_results)
            print(f'FIT_DIAGNOSIS {family} {state_name} step={step}', flush=True)
        write_json(out/f'{family}.json', family_results)
        results[family] = family_results
        del model
        torch.cuda.empty_cache()
    write_json(out/'results.json', results)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    keys = ['training_fitting_stations', 'training_withheld_stations',
            'withheld_weather_fitting_stations', 'withheld_weather_withheld_stations']
    labels = ['Fit hours / fit stations', 'Fit hours / withheld stations',
              'Withheld weather / fit stations', 'Withheld weather / withheld stations']
    for col, (family, values) in enumerate(results.items()):
        for key, label in zip(keys, labels):
            for row, metric in enumerate(('macro_rmse', 'CRPS_mm')):
                chosen = ['initial', 'point_best', 'last']
                axes[row, col].plot(range(3), [values[c]['panels'][key][metric] for c in chosen], 'o-', label=label)
                axes[row, col].set_xticks(range(3), ['Initial', 'Point selected', 'Final'])
                axes[row, col].set(title=family.upper(), ylabel='Macro RMSE (mm)' if row == 0 else 'Micro CRPS (mm)')
                axes[row, col].grid(alpha=.2)
        axes[0, col].legend(fontsize=7)
    fig.savefig(out/'fixed_panel_fit.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, family in zip(axes, results):
        curve = read_json(root/family/'curve.json')
        records = [v for v in curve if v.get('losses') and v['phase'] == 'observation_proper_scores']
        steps = [v['step'] for v in records]
        ax.plot(steps, [v['losses']['es_mm'] for v in records], 'o-', label='Training ES (changing batches)')
        ax.plot(steps, [v['validation']['ES_mm'] for v in records], 'o-', label='Fixed withheld ES')
        ax.set(title=family.upper(), xlabel='Updates', ylabel='ES (mm)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(out/'optimization_vs_validation.png', dpi=180)
    plt.close(fig)
    lines = ['# 三类模型短训：优化与拟合诊断', '',
             '固定24个训练时刻与24个内部留出天气时刻，分别评价拟合站和内部留出站。自由先验采样K8，噪声前缀固定。',
             '这是训练完成后的诊断，不据此重新选模；本脚本不评分C/D。不同站群的绝对误差不能直接解释为过拟合间隙，应比较同一人口的前后变化。', '',
             '|模型|权重|步数|训练时刻/拟合站RMSE|训练时刻/留出站RMSE|留出天气/拟合站RMSE|留出天气/留出站RMSE|',
             '|---|---|---:|---:|---:|---:|---:|']
    for family, values in results.items():
        for name, value in values.items():
            scores = '|'.join(f"{value['panels'][k]['macro_rmse']:.4f}" for k in keys)
            lines.append(f"|{family}|{name}|{value['step']}|{scores}|")
    lines += ['', '读取方法：固定训练人口评分下降说明模型可以拟合；若同一留出人口先改善后变差，同时固定训练人口继续改善，支持该训练阶段过拟合。',
              '若训练和留出均无改善，应进一步检查优化、损失目标和输入信息。单凭随机批损失或一次K8采样不作收敛结论。',
              'VAE ELBO预训练损失与后续ES目标量纲不同，图中仅比较观测评分阶段；GAN对抗项不与其他模型总损失直接排名。', '']
    (out/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(out/'complete.json', dict(status='COMPLETE', seconds=time.monotonic()-train.CLOCK_START,
                                       results_sha256=sha256(out/'results.json')))
    train.commit_budget('COMPLETE_WITH_FIT_DIAGNOSIS')
    print('FIT_DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    with train.single_process():
        try:
            main()
        except Exception:
            if train.PRIOR_SPENT > 0:
                train.commit_budget('FAILED_FIT_DIAGNOSIS')
            raise
