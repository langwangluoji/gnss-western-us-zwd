"""One-time evaluation-only migration; retain every trained tensor and cursor."""
import shutil
from pathlib import Path
import numpy as np
import torch
from gnss_field.artifacts import immutable_json, read_json, write_json, sha256
from train_independent_generators import ROOT, PROJECT, science_lock
from train_enhanced_encoder import atomic_checkpoint


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            same_tree(left, right)
    else:
        assert a == b


def main():
    old = PROJECT/'results/20261123_enhanced_encoder_generative_v02'
    assert (old/'PAUSE').exists() and read_json(old/'status.json')['stage'] == 'PAUSED'
    assert not ROOT.exists(), 'Migration destination already exists; inspect rather than overwrite'
    previous, current = read_json(old/'generators.lock.json'), science_lock()
    assert {k for k in previous if previous[k] != current.get(k)} == {'training_code'}
    assert set(current)-set(previous) == {'spatial_evaluation'}
    snapshot = (old/'release_snapshot/train_independent_generators.py').read_text(encoding='utf-8')
    assert sha256(old/'release_snapshot/train_independent_generators.py') == previous['training_code']
    changed = (PROJECT/'tools/train_independent_generators.py').read_text(encoding='utf-8')
    restored = changed.replace('generative_v03', 'generative_v02')
    restored = restored.replace("\n\ndef fixed_spatial_rows(original_rows):\n    rows = np.asarray(original_rows, int)\n    return rows[np.rint(np.linspace(0, len(rows)-1, 48)).astype(int)]\n", '')
    restored = restored.replace("                spatial_evaluation='fixed48 original unfiltered training-calendar rows; independent of training qualification',\n", '')
    restored = restored.replace("fixed = fixed_spatial_rows(data.rows('train'))", "fixed = allowed[np.rint(np.linspace(0, len(allowed)-1, 48)).astype(int)]")
    assert restored == snapshot, 'Unexpected change to training path; migration is forbidden'
    ROOT.mkdir()
    immutable_json(ROOT/'generators.lock.json', current)
    new_contract = sha256(ROOT/'generators.lock.json')
    old_contract = sha256(old/'generators.lock.json')
    records = []
    for family, expected_step in [('vae', 1800), ('gan', 1600)]:
        destination = ROOT/family
        destination.mkdir()
        source = old/family
        for name in ('last.pt', 'point_best.pt', 'joint_best.pt'):
            state = torch.load(source/name, map_location='cpu', weights_only=False)
            assert state['family'] == family and state['contract_sha256'] == old_contract
            if name == 'last.pt':
                assert state['step'] == expected_step
            original_sha = sha256(source/name)
            migrated = dict(state, contract_sha256=new_contract,
                            migration=dict(source_checkpoint_sha256=original_sha,
                                           reason='Restore original fixed48 spatial evaluation calendar; training unchanged'))
            atomic_checkpoint(destination/name, migrated)
            reloaded = torch.load(destination/name, map_location='cpu', weights_only=False)
            unchanged = {k: v for k, v in reloaded.items() if k != 'migration'}
            unchanged['contract_sha256'] = old_contract
            same_tree(state, unchanged)
            records.append(dict(family=family, checkpoint=name, step=state['step'],
                                old_sha256=original_sha, new_sha256=sha256(destination/name),
                                all_training_state_equal=True))
        for name in ('curve.json', 'events.jsonl', 'elbo_pretrained.pt'):
            if (source/name).exists():
                shutil.copy2(source/name, destination/name)
        # Only the spatial-only calendar changed. Verify and retain other ensembles.
        for product in ('point', 'joint'):
            for cohort in ('temporal', 'spatial_temporal', 'C2019_development'):
                name = f'{product}_{cohort}'
                if not (source/f'{name}.json').exists():
                    continue
                receipt = read_json(source/f'{name}.json')
                assert receipt['checkpoint_sha256'] == sha256(source/f'{product}_best.pt')
                assert receipt['artifact_sha256'] == sha256(source/f'{name}.npz')
                assert receipt['data_sha256'] == current['input_cache'] and receipt['members'] == 32
                shutil.copy2(source/f'{name}.npz', destination/f'{name}.npz')
                receipt['migration'] = dict(source_receipt_sha256=sha256(source/f'{name}.json'),
                                           old_checkpoint_sha256=receipt['checkpoint_sha256'],
                                           tensor_identity_verified=True, evaluation_population_unchanged=True)
                receipt['checkpoint_sha256'] = sha256(destination/f'{product}_best.pt')
                write_json(destination/f'{name}.json', receipt)
    for name in ('training_qualification.json', 'local_budget.json', 'ARCHITECTURE.md', 'RUNBOOK.md'):
        shutil.copy2(old/name, ROOT/name)
    for name in ('ARCHITECTURE.md', 'RUNBOOK.md'):
        path = ROOT/name
        content = path.read_text(encoding='utf-8').replace('generative_v02', 'generative_v03').replace('当前有效生成批次v02', '当前有效生成批次v03')
        path.write_text(content, encoding='utf-8')
    with (ROOT/'RUNBOOK.md').open('a', encoding='utf-8') as stream:
        stream.write('\n\nv03修复空间48小时的选取：从未过滤的864小时原始日历固定选择。v02已训练的VAE1800步、GAN1600步全部参数、优化器、随机状态和游标逐项验证一致后迁移。原空间评价重新计算，其他人口经核验后保留。累计预算原样继承，不重复训练。见migration.json。\n')
    release = ROOT/'release_snapshot'
    release.mkdir()
    for path in (old/'release_snapshot').iterdir():
        shutil.copy2(path, release/path.name)
    shutil.copy2(PROJECT/'tools/train_independent_generators.py', release/'train_independent_generators.py')
    write_json(ROOT/'migration.json', dict(source=str(old), destination=str(ROOT), checkpoints=records,
               scientific_changes=['spatial-only fixed48 uses original864 calendar'],
               consumed_seconds_carried=read_json(old/'local_budget.json')['consumed_seconds']))
    write_json(old/'superseded.json', dict(replacement=str(ROOT), reason='Spatial evaluation calendar correction only; weights retained'))
    write_json(ROOT/'status.json', dict(stage='MIGRATED_READY', next='correct spatial evaluation, finish GAN evaluation, train diffusion'))
    print('MIGRATION_VERIFIED', len(records), 'checkpoints; cumulative budget preserved')


if __name__ == '__main__':
    main()
