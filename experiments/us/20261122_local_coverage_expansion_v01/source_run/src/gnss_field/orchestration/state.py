"""Per-task single-writer events and atomic result receipts."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import os

from ..artifacts import canonical, fingerprint, immutable_json, read_json, sha256, write_json
from .campaign import Task

TASK_STATES = {'RUNNING', 'PAUSED', 'FAILED', 'SUCCEEDED', 'STOPPED_NO_GAIN'}


@contextmanager
def single_owner(path):
    """Use an OS-held lock that releases on process exit, including an unclean worker crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, 'r+b', buffering=0) as handle:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b'\0')
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f'An active owner lock exists: {path}') from error
        try:
            handle.seek(0)
            handle.write(str(os.getpid()).encode('ascii'))
            handle.flush()
            os.fsync(handle.fileno())
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    # Keep the inode stable for contenders that opened the same lock path.


def set_task_state(directory, task: Task, state: str, *, detail='', attempt=None):
    if state not in TASK_STATES:
        raise ValueError(f'Unknown task state {state}')
    root = Path(directory) / 'results' / task.task_id
    root.mkdir(parents=True, exist_ok=True)
    event = {'state': state, 'task_id': task.task_id, 'run_id': task.run_id,
             'task_hash': fingerprint(task.to_dict()), 'detail': str(detail),
             'time': datetime.now(timezone.utc).isoformat(), 'attempt': attempt}
    with single_owner(root / '.state.lock'):
        with (root / 'events.jsonl').open('ab') as handle:
            handle.write(canonical(event) + b'\n')
            handle.flush()
            os.fsync(handle.fileno())
        write_json(root / 'status.json', event)
    return event


def commit_task_result(directory, task: Task, outputs: dict, *, gate='PASS') -> dict:
    """Commit a successful result only after every declared output exists and hashes successfully."""
    if gate != 'PASS':
        raise ValueError('A successful receipt requires gate PASS')
    root = Path(directory) / 'results' / task.task_id
    files = {}
    for name, output in outputs.items():
        path = Path(output).resolve()
        if not path.is_file():
            raise ValueError(f'Missing output artifact: {path}')
        files[name] = {'path': str(path), 'sha256': sha256(path)}
    receipt = {'task_id': task.task_id, 'run_id': task.run_id,
               'task_hash': fingerprint(task.to_dict()), 'state': 'SUCCEEDED', 'gate': gate,
               'artifacts': files}
    immutable_json(root / 'success.json', receipt)
    set_task_state(directory, task, 'SUCCEEDED')
    return receipt


def verified_receipt(directory, task: Task) -> dict:
    root = Path(directory) / 'results' / task.task_id
    if not (root / 'status.json').is_file() or not (root / 'success.json').is_file():
        raise ValueError(f'Missing successful artifact receipt for {task.task_id}')
    status = read_json(root / 'status.json')
    receipt = read_json(root / 'success.json')
    if status.get('state') != 'SUCCEEDED' or receipt.get('state') != 'SUCCEEDED' or receipt.get('gate') != 'PASS':
        raise ValueError(f'Task {task.task_id} is paused, failed or stopped at a scientific gate')
    if any(value.get('task_hash') != fingerprint(task.to_dict()) or value.get('run_id') != task.run_id
           for value in (status, receipt)):
        raise ValueError(f'Task artifact identity differs: {task.task_id}')
    for artifact in receipt.get('artifacts', {}).values():
        path = Path(artifact['path'])
        if not path.is_file() or sha256(path) != artifact['sha256']:
            raise ValueError(f'Task artifact hash differs: {task.task_id}')
    return receipt
