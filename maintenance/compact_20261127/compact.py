"""Read-only inventory, model archive preparation and checked deletion ledger.

This module never deletes files. Execution is confined to retire_files.ps1.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'maintenance/compact_20261127'
ROOTS = {'project': ROOT, 'cache_d': Path('D:/SITP_cache')}
R = 'artifacts/legacy_layout/research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server/results'
CORE = {
    '20261123_terrain_diffusion_v10_data', '20261125_temporal_continuous_data',
    '20261125_p14_continuous_fusion', '20261125_observed_innovation_precision',
    '20261125_innovation_calendar15', '20261126_native_weather_cache_v2',
    '20261126_native_weather_means', '20261126_within_forecast_increment',
    '20261126_directed_lag_v2', '20261126_best_base_diffusion_fusion',
}
MODEL = {'.pt', '.pth', '.ckpt', '.ubj', '.pkl', '.joblib', '.safetensors'}
ARRAY = {'.npz', '.npy', '.h5', '.hdf5'}
EVIDENCE = {'.json', '.jsonl', '.csv', '.tsv', '.txt', '.md', '.svg', '.png', '.pdf'}


def extended(path):
    path = Path(path).absolute()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def sha(path):
    with extended(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def safe_path(root, relative):
    rel = PurePosixPath(relative.replace('\\', '/'))
    if rel.is_absolute() or '..' in rel.parts or ':' in str(rel) or not rel.parts:
        raise ValueError('path escapes allowed root')
    root = Path(root).resolve()
    target = root.joinpath(*rel.parts)
    if not target.resolve().is_relative_to(root) or target == root:
        raise ValueError('path escapes allowed root')
    for parent in [target, *target.parents]:
        if parent == root:
            break
        p = extended(parent)
        if p.exists() and getattr(p.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError('reparse boundary')
    return extended(target)


def classify(scope, path, size=None):
    p = PurePosixPath(path)
    # Previous rejected recursive cache deletions are outside this action.
    if any(x in {'build', '__pycache__', '.pytest_cache', '.superpowers'} for x in p.parts):
        return 'keep_other', 'installer/test/build cache outside retirement scope'
    if scope == 'cache_d':
        if path.endswith('train_reference_utc.h5.partial'):
            return 'retire_derived', 'incomplete intermediate; completed frozen rebuild retained'
        if p.suffix in {'.h5', '.hdf5'} and (
            path.startswith('europe_dense_zwd_20261104_v1/physics/') or
            path.startswith('europe_dense_zwd_20261104_v1/validation/') or
            path.startswith('europe_ref_gan_20261103_v1/expanded_prepared/')
        ):
            return 'retire_derived', 'retired prepared data; final frozen data and dense ERA5 sources retained'
        if p.suffix in {'.h5', '.hdf5', '.parquet'}:
            return 'keep_raw', 'old final frozen source fallback or inexpensive observations'
        return 'keep_evidence', 'old model replay package and metadata retained'
    if path.startswith('artifacts/campaigns/europe_'):
        return 'keep_core', 'complete European fixed-CNN/VMF3 run and inference dependencies'
    if path.startswith('artifacts/legacy_layout/research_pipeline/data/') or '/02_data/' in path:
        return 'keep_raw', 'observations, terrain, geometry or original prepared inputs'
    if path.startswith(R + '/'):
        tail = path[len(R) + 1:]
        if tail.split('/')[0] in CORE:
            return 'keep_core', 'US I1/joint/controls/probability dependency closure'
        if p.suffix in MODEL:
            return 'archive_model', 'retired unique model or optimizer state; remote archive required'
        if p.suffix in {'.npz','.npy'} and any(x in p.stem.lower() for x in (
            'coefficient', 'parameter', 'weight', 'model', 'normaliz', 'statistic',
            'scale', 'scaler', 'basis', 'operator', 'whiten', 'calibration')):
            return 'archive_model', 'potential fitted parameters preserved rather than inferred from extension'
        if p.suffix in {'.npz','.npy'} and size is not None and size<=4*2**20:
            return 'archive_model', 'small retired arrays conservatively preserved with fitted model context'
        if p.suffix in ARRAY:
            return 'retire_derived', 'retired cache/embedding/evaluation; lightweight evidence retained'
        if p.suffix in EVIDENCE:
            return 'keep_evidence', 'protocol, report, metric, figure or trace; retained locally and archived'
    return 'keep_other', 'source code, compressed recovery package or unclassified material retained'


def scan():
    rows = []
    for scope, base in ROOTS.items():
        walk = extended(base)
        for cur, dirs, files in os.walk(walk, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in {'.git', '.superpowers'} and not (
                (Path(cur)/d).lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT))
            for name in sorted(files):
                f = Path(cur)/name
                rel = f.relative_to(walk).as_posix()
                if scope == 'project' and rel.startswith('maintenance/compact_20261127/'):
                    continue
                if f.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    continue
                action, reason = classify(scope, rel, f.stat().st_size)
                rows.append(dict(scope=scope, path=rel, size=f.stat().st_size,
                                 sha256=sha(f), action=action, reason=reason))
                if len(rows) % 500 == 0:
                    print(json.dumps({'hashed':len(rows), 'gib':round(sum(x['size'] for x in rows)/2**30,2)}), flush=True)
    write_csv(HERE/'inventory.csv', rows)
    summary(rows, 'classification.json')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['scope','path','size','sha256','action','reason'])
        w.writeheader(); w.writerows(rows)


def read_rows():
    with (HERE/'inventory.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['size'] = int(row['size'])
    return rows


def reclassify():
    rows=read_rows()
    for r in rows:
        r['action'],r['reason']=classify(r['scope'],r['path'],r['size'])
    write_csv(HERE/'inventory.csv',rows)
    summary(rows,'classification.json')


def summary(rows, name):
    groups = collections.defaultdict(lambda: {'files':0,'bytes':0})
    for r in rows:
        g = groups[r['scope']+':'+r['action']]; g['files'] += 1; g['bytes'] += int(r['size'])
    result = {'groups':dict(groups), 'total_bytes':sum(int(r['size']) for r in rows)}
    (HERE/name).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


def package():
    rows = read_rows(); out = HERE/'staging'; out.mkdir(exist_ok=True)
    # Content-addressed model objects avoid storing identical checkpoints twice.
    unique = {}
    for r in rows:
        if r['action'] == 'archive_model':
            unique.setdefault(r['sha256'], r)
    batches, batch, size = [], [], 0
    for r in unique.values():
        if batch and size + r['size'] > 700*2**20:
            batches.append(batch); batch=[]; size=0
        batch.append(r); size += r['size']
    if batch:
        batches.append(batch)
    index, assets = {}, []
    for i, batch in enumerate(batches, 1):
        name = f'retired-models-{i:02}.tar.gz'; dest=out/name
        with tarfile.open(extended(dest), 'w:gz', compresslevel=1) as tar:
            for r in batch:
                path = safe_path(ROOTS[r['scope']], r['path'])
                if sha(path) != r['sha256']:
                    raise ValueError('model changed since inventory')
                member = 'objects/'+r['sha256']+PurePosixPath(r['path']).suffix
                tar.add(path, arcname=member, recursive=False)
                index[r['sha256']] = dict(asset=name, member=member)
        # Test actual recovery bytes, not just an archive directory listing.
        with tarfile.open(extended(dest), 'r:gz') as tar:
            for member in tar:
                with tar.extractfile(member) as f:
                    recovered = hashlib.file_digest(f, 'sha256').hexdigest()
                if recovered != PurePosixPath(member.name).stem:
                    raise ValueError('model archive member hash mismatch')
        assets.append(dict(name=name, size=dest.stat().st_size, sha256=sha(dest)))
        print(json.dumps(assets[-1]), flush=True)
    # All small retired-run evidence, including untracked logs and cache receipts.
    name='retired-evidence.tar.gz'; dest=out/name
    evidence=[r for r in rows if (r['scope']=='cache_d' or r['path'].startswith(R+'/'))
              and r['action'] in {'keep_evidence','keep_other'}
              and PurePosixPath(r['path']).suffix in EVIDENCE
              and not any(t in r['path'].lower() for t in ('credential','secret','.env','token'))]
    with tarfile.open(extended(dest), 'w:gz', compresslevel=1) as tar:
        for r in evidence:
            path=safe_path(ROOTS[r['scope']],r['path'])
            if sha(path)!=r['sha256']:
                raise ValueError('evidence changed since inventory')
            tar.add(path, arcname=r['scope']+'/'+r['path'], recursive=False)
        tar.add(HERE/'inventory.csv', arcname='inventory.csv', recursive=False)
        tar.add(HERE/'PLAN.md', arcname='PLAN.md', recursive=False)
    assets.append(dict(name=name,size=dest.stat().st_size,sha256=sha(dest)))
    result={'models':index,'assets':assets,'evidence_files':len(evidence)}
    manifest=out/'recovery_manifest.json'
    manifest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    assets.append(dict(name=manifest.name,size=manifest.stat().st_size,sha256=sha(manifest)))
    (HERE/'archive_index.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'unique_models':len(index),'assets':len(assets),'evidence':len(evidence)}),flush=True)


def deletion_rows(rows, verified_models):
    result=[]
    for r in rows:
        if r['action'] == 'archive_model':
            if not verified_models.get(r['sha256'],{}).get('verified_remote'):
                raise ValueError('remote archive not verified')
            result.append(r)
        elif r['action']=='retire_derived':
            result.append(r)
    return result


def prepare_delete():
    rows=read_rows(); remote=json.loads((HERE/'remote_archive.json').read_text(encoding='utf-8'))
    plan=deletion_rows(rows,remote['models']); write_csv(HERE/'delete_plan.csv',plan)
    summary(plan,'delete_plan_summary.json')
    receipt={'inventory_sha256':sha(HERE/'inventory.csv'),'delete_plan_sha256':sha(HERE/'delete_plan.csv'),
             'archive_receipt_sha256':sha(HERE/'remote_archive.json'),'roots':{k:str(v) for k,v in ROOTS.items()}}
    (HERE/'delete_authorization.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command', choices=['scan','reclassify','package','prepare-delete']);args=parser.parse_args()
    {'scan':scan,'reclassify':reclassify,'package':package,'prepare-delete':prepare_delete}[args.command]()
