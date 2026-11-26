"""Local-only read-only monitor, optionally consuming an SSH status snapshot."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import re
import shlex
import subprocess
import threading
import time

from gnss_field.artifacts import write_json


def sync_snapshot(directory, host, remote_root):
    """Fetch a small read-only snapshot; SSH authentication is managed by the user."""
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', host):
        raise ValueError('Use a literal SSH host or user@host alias')
    payload = ('import json; from gnss_field.monitor import collect_status; '
               f'print(json.dumps(collect_status({str(remote_root)!r}),allow_nan=False))')
    command = 'python3 -c '+shlex.quote(payload)
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, command],
                                capture_output=True, text=True, encoding='utf-8', timeout=15)
        if result.returncode:
            raise RuntimeError(result.stderr[-1000:])
        snapshot = json.loads(result.stdout)
        if not snapshot.get('read_only') or not isinstance(snapshot.get('runs'), list):
            raise ValueError('Invalid monitor snapshot')
        write_json(Path(directory)/'remote_snapshot.json', snapshot)
        state = dict(sync_ok=True, last_success=time.time())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        state = dict(sync_ok=False, last_attempt=time.time(), error=str(error))
    write_json(Path(directory)/'sync_status.json', state)
    return state


def _tail(path, count=300):
    # Training logs are sparse (one row/100 updates); cap client payload explicitly.
    rows = []
    try:
        for line in path.read_text(encoding='utf-8').splitlines()[-count:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # an unfinished append is not an interpolated observation
    except FileNotFoundError:
        pass
    return rows


def collect_status(directory):
    root = Path(directory).resolve()
    snapshot = root/'remote_snapshot.json'
    if snapshot.exists():
        result = json.loads(snapshot.read_text(encoding='utf-8'))
        result['served_at'] = time.time()
        sync_file = root/'sync_status.json'
        if sync_file.exists():
            result['ssh_sync'] = json.loads(sync_file.read_text(encoding='utf-8'))
        return result
    runs = []
    for log in sorted(root.rglob('training.jsonl')):
        folder = log.parent
        status = folder/'status.json'
        try:
            state = json.loads(status.read_text(encoding='utf-8')) if status.exists() else {}
        except json.JSONDecodeError:
            state = {}
        runs.append(dict(name=folder.relative_to(root).as_posix(), status=state,
                         training=_tail(log), development=_tail(folder/'development.jsonl')))
    latest = max((r['status'].get('updated', 0) for r in runs), default=None)
    return dict(root=str(root), generated_at=time.time(), last_update=latest, runs=runs,
                read_only=True, no_interpolated_metrics=True)


def serve(directory, port=8767, *, ssh_host=None, remote_root=None):
    root = Path(directory).resolve()
    if bool(ssh_host) != bool(remote_root):
        raise ValueError('SSH monitoring requires both host and remote root')
    if ssh_host:
        def synchronize():
            while True:
                sync_snapshot(root, ssh_host, remote_root)
                time.sleep(10)
        threading.Thread(target=synchronize, daemon=True).start()
    page = (Path(__file__).parent/'resources/monitor.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/status':
                data = json.dumps(collect_status(root), ensure_ascii=False, allow_nan=False).encode('utf-8')
                content_type = 'application/json; charset=utf-8'
            elif self.path in ('/', '/index.html'):
                data, content_type = page, 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *_):
            pass
    print(f'Read-only monitor: http://127.0.0.1:{port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
