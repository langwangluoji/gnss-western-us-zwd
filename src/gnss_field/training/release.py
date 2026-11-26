"""Verify all published dependencies before benchmarking, training or evaluation."""
import hashlib
import json
from pathlib import Path

REQUIRED_FILES = {
    'src/gnss_field/training/observable.py', 'src/gnss_field/training/observable_evaluate.py',
    'src/gnss_field/training/engine.py', 'src/gnss_field/training/release.py',
    'src/gnss_field/models/subspace.py', 'src/gnss_field/models/mean_shift.py',
    'src/gnss_field/models/diffusion.py', 'src/gnss_field/objectives/scores.py',
    'src/gnss_field/runtime.py', 'bundle/data.npz', 'bundle/manifest.json',
    'evaluation/data.npz', 'evaluation/manifest.json',
}


def verify_release(root, *, required=True):
    root = Path(root).resolve()
    manifest = root/'release.json'
    if not manifest.is_file():
        if required:
            raise ValueError('Formal deployment requires release.json')
        return None
    raw = manifest.read_bytes()
    files = json.loads(raw)['files']
    if not isinstance(files, dict) or not REQUIRED_FILES.issubset(files):
        raise ValueError('Release manifest is empty or omits mandatory dependencies')
    for relative, expected in files.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Published dependency missing or outside release: {relative}')
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Published dependency differs: {relative}')
    return hashlib.sha256(raw).hexdigest()
