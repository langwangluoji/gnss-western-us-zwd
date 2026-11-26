"""Content-addressed immutable metadata and atomic, single-writer outputs."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import io
import json
import os
import tempfile

import numpy as np


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'), default=_json_default).encode('utf-8')


def fingerprint(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: str | Path) -> str:
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_bytes(path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_bytes(path, canonical(value) + b'\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if fingerprint(read_json(path)) != fingerprint(value):
            raise ValueError(f'Immutable artifact differs: {path}')
    else:
        write_json(path, value)


def atomic_numpy(path, array):
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    atomic_bytes(path, buf.getvalue())


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        a = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def release_digest(root):
    root = Path(root)
    folders = ['src', 'configs', 'slurm']
    if (root / 'research/engine').is_dir():
        folders.extend(['research/engine/tools', 'research/engine/configs',
                        'research/engine/slurm'])
    return fingerprint({p.relative_to(root).as_posix(): sha256(p)
                        for folder in folders
                        for p in sorted((root / folder).rglob('*'))
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
                        and not any(part.endswith('.egg-info') for part in p.parts)})


def package_digest(package_root=None):
    """Hash executing package bytes independently of source/wheel installation path."""
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    if not (root/'__init__.py').is_file():
        raise ValueError('A real gnss_field package directory is required')
    return fingerprint({p.relative_to(root).as_posix(): sha256(p)
                        for p in sorted(root.rglob('*'))
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'})
