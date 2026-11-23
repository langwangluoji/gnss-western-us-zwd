"""Read-only B2018 monitor. Never open atomically replaced training status files."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
from local_context_data import PROJECT, RUN


def collect_local_status(root=RUN):
    root = Path(root)
    runs = []
    def lines(path):
        try:
            rows = path.read_text(encoding='utf-8').splitlines()[-300:]
        except FileNotFoundError:
            return []
        result = []
        for row in rows:
            try:
                result.append(json.loads(row))
            except json.JSONDecodeError:
                pass
        return result
    for log in sorted(root.glob('fold_*/*/training.jsonl')):
        training = lines(log)
        development = lines(log.parent/'development.jsonl')
        last = training[-1] if training else {}
        complete = (log.parent/'results.json').exists()
        initial_path = log.parent/'initial_metrics.json'
        candidates = []
        if initial_path.exists():
            zero = json.loads(initial_path.read_text(encoding='utf-8'))['joint']
            candidates.append(dict(value=zero['value'], step=0))
        for row in development:
            if not row['startup']:
                candidates.append(dict(value=row['metrics']['joint']['value'], step=row['step']))
        products = {'joint':[min(candidates, key=lambda r:(r['value'],r['step']))]} if candidates else {}
        state = dict(stage='B2018 local study', status='COMPLETE' if complete else '日志已记录，等待完成标记',
            step=last.get('step',0), updated=last.get('updated'), products=products)
        runs.append(dict(name=log.parent.relative_to(root).as_posix(), status=state,
            training=training, development=development))
    return dict(root=str(root), generated_at=time.time(), runs=runs,
        last_update=max((r['status'].get('updated') or 0 for r in runs), default=None),
        read_only=True, no_interpolated_metrics=True, full2019=[])


def display_status(payload):
    result = copy.deepcopy(payload)
    result['runs'] = [r for r in result['runs'] if r['name'].startswith('fold_')]
    for run in result['runs']:
        run['status']['stage'] = 'B2018内层开发 · '+run['name']
        for row in run['development']:
            joint = row['metrics'].get('joint', {})
            if 'rmse_mm' in joint:
                row['metrics']['point'] = dict(value=joint['rmse_mm'], diagnostic_only=True)
    return result


def main():
    page = (PROJECT/'src/gnss_field/resources/monitor.html').read_text(encoding='utf-8')
    page = page.replace('开发子集', 'B2018内层开发').replace('完整2019评价', '完整2019评价（本轮尚未运行）')
    page = page.replace('工程验收样本不代表最终精度。', '本轮只按内层ES选模，RMSE为同步诊断；不代表全年精度。')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/status':
                data = json.dumps(display_status(collect_local_status()), ensure_ascii=False,
                    allow_nan=False).encode('utf-8')
                kind = 'application/json; charset=utf-8'
            elif self.path in ('/', '/index.html'):
                data, kind = page.encode('utf-8'), 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self, *_):
            pass
    print('B2018 read-only monitor: http://127.0.0.1:8778', flush=True)
    ThreadingHTTPServer(('127.0.0.1',8778), Handler).serve_forever()


if __name__ == '__main__':
    main()
