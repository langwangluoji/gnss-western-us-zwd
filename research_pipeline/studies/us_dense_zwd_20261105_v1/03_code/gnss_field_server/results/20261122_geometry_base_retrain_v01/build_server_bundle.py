"""Publish the verified, self-contained code archive; ready metadata is written last."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(2**20):
            h.update(block)
    return h.hexdigest()


def main():
    receipt = json.loads((ROOT / 'local_v05_engineering.json').read_text())
    assert receipt['passed']
    assert receipt['runner'] == digest(ROOT / 'train_new_base.py')
    assert receipt['probability'] == digest(ROOT / 'probability.lock.json')
    assert '356 passed' in (ROOT / 'tests.log').read_text()
    files = set((PROJECT / 'src/gnss_field').rglob('*.py'))
    files.update((PROJECT / 'tools').glob('*.py'))
    for name in ('pilot_data.py', 'pilot_models.py', 'run_pilot.py', 'gp_ndp.py'):
        files.add(PROJECT / 'results/20261122_local_generative_architecture_v01' / name)
    for name in ('retrain_common.py', 'retrain_geometry.py', 'retrain_data.py',
                 'train_new_base.py', 'server_submit.py', 'remote_bridge.py',
                 'test_retrain.py', 'annual_base.py', 'build_server_bundle.py',
                 'A800.sh', 'PLAN.md', 'mean.lock.json', 'probability.lock.json',
                 'probability_factors.npy', 'annual_results.json',
                 'annual_paired_intervals.json', 'ANNUAL_RESULTS.md',
                 'mean_equivalence.json', 'integration_acceptance.json',
                 'local_v05_engineering.json', 'tests.log'):
        files.add(ROOT / name)
    files.add(PROJECT / 'results/20261122_local_operator_exploration_v01/selection.lock.json')
    ordered = sorted(files)
    manifest = {'files': {p.relative_to(PROJECT).as_posix():
                         {'sha256': digest(p), 'bytes': p.stat().st_size}
                         for p in ordered}}
    payload = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    (ROOT / 'code_manifest.json').write_bytes(payload)
    archive = ROOT / 'server_code.tar.gz'
    temporary = archive.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as tar:
        for path in ordered:
            assert path.is_file() and not path.is_symlink()
            tar.add(path, arcname=path.relative_to(PROJECT).as_posix(), recursive=False)
        member = tarfile.TarInfo('code_manifest.json')
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    with tarfile.open(temporary) as tar:
        for member in tar.getmembers():
            assert member.isfile() and not Path(member.name).is_absolute()
            assert '..' not in Path(member.name).parts
            if member.name != 'code_manifest.json':
                assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == manifest['files'][member.name]['sha256']
    temporary.replace(archive)
    result = {'sha256': digest(archive), 'bytes': archive.stat().st_size,
              'files': len(manifest['files']), 'runner': receipt['runner']}
    ready = ROOT / 'server_code_bundle.json'
    ready.with_suffix('.tmp').write_text(json.dumps(result), encoding='utf-8')
    ready.with_suffix('.tmp').replace(ready)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
