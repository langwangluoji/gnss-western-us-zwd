"""Migrate one interrupted M0 run after the verified scoring-mask-only fix."""
import argparse
import copy
from dataclasses import asdict, replace
from pathlib import Path
import shutil

import numpy as np
import torch

from gnss_field.artifacts import (
    fingerprint, immutable_json, model_digest, package_digest, read_json,
    release_digest, sha256, write_json,
)
from gnss_field.cli import settings
from gnss_field.data import PackedStore
from gnss_field.data.manifests import build_training_manifest, stratified_hours
from gnss_field.models.mean import MeanField
from gnss_field.orchestration import Campaign, Task, write_campaign
from gnss_field.orchestration.budget import validate_preflight
from gnss_field.runtime import doctor, verified_release_hashes
from gnss_field.training.engine import atomic_torch
from gnss_field.training.prediction import evaluate_predictor


OLD = "                               arrays['normal_support_S0'], samples=arrays.get('samples'), std=arrays.get('std'),"
NEW = "                               arrays['normal_support_S0'] & arrays['population_F'],\n                               samples=arrays.get('samples'), std=arrays.get('std'),"


def verify_source_change(old_source, new_source):
    def files(root):
        return {p.relative_to(root).as_posix(): sha256(p)
                for folder in ('src', 'configs', 'slurm')
                for p in (root/folder).rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
                and not any(part.endswith('.egg-info') for part in p.parts)}

    old_files, new_files = files(old_source), files(new_source)
    changed = {name for name in old_files.keys() | new_files.keys()
               if old_files.get(name) != new_files.get(name)}
    target = 'src/gnss_field/training/prediction.py'
    if changed != {target}:
        raise ValueError(f'Recovery requires exactly the scoring-mask fix, got {changed}')
    old_text = (old_source/target).read_text(encoding='utf-8')
    new_text = (new_source/target).read_text(encoding='utf-8')
    if old_text.count(OLD) != 1 or old_text.replace(OLD, NEW) != new_text:
        raise ValueError('Unexpected evaluator changes')


def migrate_checkpoint(state, old_contract, science, hours, old_digest, new_digest):
    if state['step'] != 200 or not state['pending_evaluation'] or state['products']:
        raise ValueError('Expected interrupted step-200 startup evaluation')
    old_spec = state['spec']
    if old_spec['dependencies']['release_sha256'] != old_digest:
        raise ValueError('Checkpoint release differs from the verified parent')
    if fingerprint(old_contract['spec']) != fingerprint(old_spec):
        raise ValueError('Checkpoint stage differs from its saved contract')
    if fingerprint(old_contract['science']) != fingerprint(science):
        raise ValueError('Scientific configuration changed')
    expected = fingerprint(dict(spec=old_spec, science=asdict(science),
                                hours=np.asarray(hours), model='MeanField'))
    if expected != state['signature'] or expected != old_contract['signature']:
        raise ValueError('Parent training signature mismatch')
    migrated = dict(state)
    migrated['spec'] = copy.deepcopy(old_spec)
    migrated['spec']['dependencies']['release_sha256'] = new_digest
    migrated['signature'] = fingerprint(dict(spec=migrated['spec'], science=asdict(science),
                                            hours=np.asarray(hours), model='MeanField'))
    return migrated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    parent, source, root = args.parent.resolve(), args.source.resolve(), args.destination.resolve()
    science, runtime, extra = settings(argparse.Namespace(config=str(parent/'resolved.json')))
    old_source = Path(extra['project_root'])
    verify_source_change(old_source, source)
    old_hashes = read_json(parent/'campaign.json')['hashes']
    if release_digest(old_source) != old_hashes['code']:
        raise ValueError('Parent source release changed')
    old_digest = package_digest(old_source/'src/gnss_field')
    new_digest = package_digest(source/'src/gnss_field')
    extra['project_root'] = str(source)
    runtime = replace(runtime, wall_hours=7.75)
    hashes = verified_release_hashes(science, runtime, source)
    if any(hashes[k] != old_hashes[k] for k in ('plan', 'data', 'protocol')):
        raise ValueError('Recovery changed data, science or evaluation protocol')
    root.mkdir(parents=True, exist_ok=True)
    immutable_json(root/'resolved.json', dict(science=asdict(science), runtime=asdict(runtime), campaign=extra))
    task = Task('mean_base', 'mean_base', 0, ('mean_base',), (), root.name+'/mean_base',
                7.75, max_attempts=1)
    campaign = Campaign(root.name, (task,), hashes)
    write_campaign(root, campaign, runtime, environment=extra['environment'], working_directory=source)
    for name in ('training_manifest.npz', 'training_manifest.json'):
        destination = root/name
        if not destination.exists():
            shutil.copy2(parent/name, destination)
        if sha256(destination) != sha256(parent/name):
            raise ValueError('Training manifest differs')
    store = PackedStore(runtime.data_root, purpose='train')
    manifest = build_training_manifest(store, root/'training_manifest.npz')
    hours = manifest.select((2016, 2017))
    old_mean, mean = parent/'seed_42/mean_base', root/'seed_42/mean_base'
    mean.mkdir(parents=True, exist_ok=True)
    state = torch.load(old_mean/'last.pt', map_location='cpu', weights_only=False)
    old_contract = read_json(old_mean/'contract.json')
    migrated = migrate_checkpoint(state, old_contract, science, hours, old_digest, new_digest)
    receipt = root/'recovery.json'
    if not receipt.exists():
        if (mean/'last.pt').exists():
            raise ValueError('Partial migration requires inspection')
        atomic_torch(mean/'last.pt', migrated)
        immutable_json(mean/'contract.json', dict(signature=migrated['signature'],
                                                 spec=migrated['spec'], science=science))
        shutil.copy2(old_mean/'training.jsonl', mean/'training.jsonl')
        shutil.copytree(old_mean/'eval_000200', mean/'eval_000200', dirs_exist_ok=True)
        immutable_json(receipt, dict(parent=str(parent), parent_checkpoint_sha256=sha256(old_mean/'last.pt'),
                       migrated_checkpoint_sha256=sha256(mean/'last.pt'),
                       old_signature=state['signature'], new_signature=migrated['signature'],
                       old_hashes=old_hashes, new_hashes=hashes, step=200,
                       change='Score geometric support intersected with valid observation population',
                       carried_state=['model', 'ema', 'optimizers', 'rng', 'cursor', 'pending_evaluation']))
    model = MeanField()
    model.load_state_dict(state['ema'])
    development = PackedStore(runtime.data_root, purpose='development')
    evaluation_hours = stratified_hours(np.asarray(development.hours), 2019)
    def cached_only(*args, **kwargs):
        raise ValueError('Startup prediction cache is incomplete; GPU evaluation required')
    torch.set_num_threads(2)
    result = evaluate_predictor(development, evaluation_hours, cached_only, mean/'eval_000200',
                               members=science.screen_members, checkpoint_hash=model_digest(model),
                               use_history=science.use_history)
    value = result['values']['point']['value']
    if not np.isfinite(value):
        raise ValueError('Repaired development RMSE is nonfinite')
    print('RECOVERED_STARTUP', result['metrics']['primary'], flush=True)
    probe = read_json(parent/'preflight.json')
    preflight = doctor(runtime, live_tongji=True, hashes=hashes)
    for key in ('driver_ok', 'gpu_probe_hostname', 'actual_gpu', 'actual_memory_gb'):
        preflight[key] = probe[key]
    preflight['driver_evidence_parent'] = str(parent/'preflight.json')
    preflight['recovery_receipt_sha256'] = sha256(receipt)
    validate_preflight(preflight, campaign, runtime)
    write_json(root/'preflight.json', preflight)
    immutable_json(root/'budget.lock.json', dict(locked=True, hashes=hashes,
        allocated_profile='tongji_a800', total_card_hours=7.75,
        stage_card_hours={'mean_base': 7.75}, gpu_unit_CNY=4,
        parent_benchmark_sha256=read_json(parent/'budget.lock.json')['benchmark_sha256'],
        recovery_receipt_sha256=sha256(receipt),
        reason='Evaluation-only fix; remaining part of the original eight-hour allocation budget'))
    print('RECOVERY_READY', root, flush=True)


if __name__ == '__main__':
    main()
