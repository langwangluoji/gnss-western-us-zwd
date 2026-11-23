"""Read-only localhost monitor, reusing the existing log-server pattern."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from field_generation_v1.prepare import ROOT

def read(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):return None

def snapshot():
    result=dict(status=read(ROOT/'status.json'),arms={})
    for arm in ['M0','M2-F','M2-J']:
        directory=ROOT/arm;records={}
        if (directory/'training.jsonl').exists():
            for line in (directory/'training.jsonl').read_text().splitlines():
                try:r=json.loads(line);records[r['step']]=r
                except (ValueError,KeyError):pass
        result['arms'][arm]=dict(training=sorted(records.values(),key=lambda r:r['step']),
            development=[r for p in sorted(directory.glob('development_*.json')) if (r:=read(p))],
            selected=read(directory/'selected.json'),full2019=read(directory/'full2019.json'))
    return result

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?')[0]=='/state':body=json.dumps(snapshot(),allow_nan=False).encode();kind='application/json'
        elif self.path=='/':body=Path(__file__).with_name('monitor.html').read_bytes();kind='text/html'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*a):pass

if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()
