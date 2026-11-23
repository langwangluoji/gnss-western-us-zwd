"""Create a code-only portable ZIP and optional separate immutable data archive."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def archive(target, root, paths):
    records = {}
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(paths):
            name = path.relative_to(root).as_posix()
            z.write(path, name)
            records[name] = sha(path)
        z.writestr('ARCHIVE_SHA256.json', json.dumps(records, sort_keys=True, indent=2))
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
        for name, expected in records.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError(f'Archive content hash mismatch: {name}')
    return {'path': str(target.resolve()), 'sha256': sha(target), 'bytes': target.stat().st_size,
            'members': len(records), 'crc_and_hashes_verified': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='dist')
    p.add_argument('--data')
    p.add_argument('--skip-wheel-build', action='store_true', help='Package an already built and verified wheel')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_wheel_build:
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(root), '--no-deps',
                        '--no-build-isolation', '-w', str(output)], check=True)
    elif not list(output.glob('*.whl')):
        raise ValueError('No verified wheel is available to package')
    paths = [root/'pyproject.toml', root/'README.md']
    for folder in ('src', 'configs', 'slurm', 'docs', 'tests', 'tools'):
        paths += [f for f in (root/folder).rglob('*') if f.is_file() and
                  '__pycache__' not in f.parts and not any(x.endswith('.egg-info') for x in f.parts)]
    for name in ('FINAL_VALIDATION.json', 'environment.json', 'final_tests.xml', 'wheel_validation.json',
                 'release_cuda/smoke.json', 'final_audit/data_audit.json', 'profile_kernels/smoke.json',
                 'data_pack_report.json', 'training_population.json', 'data_archive.json'):
        evidence = root/'local_validation'/name
        if evidence.is_file():
            paths.append(evidence)
    # Add the installable wheel alongside source, with a stable relative path.
    with zipfile.ZipFile(output/'gnss_field_server_code.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        records = {}
        for path in sorted(paths):
            name = 'gnss_field_server/'+path.relative_to(root).as_posix()
            z.write(path, name)
            records[name] = sha(path)
        for path in output.glob('*.whl'):
            name = 'gnss_field_server/dist/'+path.name
            z.write(path, name)
            records[name] = sha(path)
        z.writestr('ARCHIVE_SHA256.json', json.dumps(records, sort_keys=True, indent=2))
    code = output/'gnss_field_server_code.zip'
    with zipfile.ZipFile(code) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(k)).hexdigest() == v for k,v in records.items())
    report = {'code': {'path': str(code), 'sha256': sha(code), 'bytes': code.stat().st_size,
                       'members': len(records), 'crc_and_hashes_verified': True}}
    if args.data:
        data = Path(args.data).resolve()
        if not (data/'manifest.json').is_file():
            raise ValueError('Data archive requires a packed dataset manifest')
        report['data'] = archive(output/'gnss_field_server_data.zip', data.parent,
                                 [f for f in data.rglob('*') if f.is_file()])
    (output/'DELIVERY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
