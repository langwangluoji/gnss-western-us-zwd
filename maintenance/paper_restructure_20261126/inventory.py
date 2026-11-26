"""Read-only content inventory and bounded duplicate proofs; never deletes files."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('target must be strictly inside project')
    return path


def sha(path):
    path = Path(path).resolve()
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        path = Path('\\\\?\\' + str(path))
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def duplicate_proof(candidate, reference):
    candidate, reference = Path(candidate), Path(reference)
    if candidate.resolve() == reference.resolve():
        raise ValueError('reference must survive separately')
    digest = sha(candidate)
    if candidate.stat().st_size != reference.stat().st_size or digest != sha(reference):
        raise ValueError('contents differ')
    return dict(candidate=str(candidate), surviving_reference=str(reference),
                size=candidate.stat().st_size, sha256=digest)


def inventory(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    walk_root = Path('\\\\?\\' + str(root)) if os.name == 'nt' else root
    rows, total, start = [], 0, time.monotonic()
    for current, dirs, files in os.walk(walk_root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '.superpowers', '__pycache__', '.pytest_cache'} and not d.endswith('.egg-info')
                         and not (Path(current) / d).is_symlink())
        for name in sorted(files):
            path = Path(current) / name
            relative = path.relative_to(walk_root).as_posix()
            if relative == output.relative_to(root).as_posix() or path.is_symlink() or path.name.endswith('.local.txt'):
                continue
            info = path.stat()
            rows.append(dict(path=relative, size=info.st_size, sha256=sha(path)))
            total += info.st_size
            if len(rows) % 1000 == 0:
                print(json.dumps(dict(files=len(rows), gib=round(total / 2**30, 3), seconds=round(time.monotonic()-start))), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'size', 'sha256']); writer.writeheader(); writer.writerows(rows)
    receipt = dict(files=len(rows), bytes=total, manifest_sha256=sha(output), seconds=round(time.monotonic()-start),
                   policy='read-only byte hashing; no label parsing; Python/test caches and git internals excluded')
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=Path.cwd()); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); inventory(args.root, args.output)
