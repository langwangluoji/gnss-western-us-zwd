"""Local read-only paired-study monitor; append-only telemetry, no training controls."""
import argparse
import copy
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import time
from grid_pair_data import RUN,PROJECT
from local_context_monitor import collect_local_status


def monitor_payload(payload):
    result=copy.deepcopy(payload)
    for run in result['runs']:
        run['status']['stage']='B2018内层开发 · '+run['name']
        for row in run['development']:
            joint=row['metrics'].get('joint',{})
            if 'rmse_mm' in joint:
                row['metrics']['point']=dict(value=joint['rmse_mm'],diagnostic_only=True)
    return result


def read_remote_status(path):
    path=Path(path)
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-2_000_000))
        rows=stream.read().decode('utf-8',errors='replace').splitlines()
    for row in reversed(rows):
        if not row.startswith('GRID_SNAPSHOT '):
            continue
        try:
            payload=json.loads(row[len('GRID_SNAPSHOT '):])
        except json.JSONDecodeError:
            continue
        last=payload['generated_at']
        payload['ssh_sync']=dict(sync_ok=time.time()-last<120,last_success=last,
            last_attempt=time.time(),error='' if time.time()-last<120 else '连接暂无新快照，保留最后记录')
        return payload
    return None


def stage_status(engine_state,has_results,slurm_state):
    if has_results: return 'COMPLETE'
    if engine_state in ('PAUSED','BUDGET_EXHAUSTED','PAUSED_DURING_EVALUATION'): return engine_state
    if slurm_state in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'): return slurm_state
    if slurm_state in ('PENDING','SUSPENDED'): return slurm_state
    if engine_state=='COMPLETE': return 'OUTER_EVALUATION'
    return slurm_state or engine_state or 'WAITING'


def build_page():
    page=(PROJECT/'src/gnss_field/resources/monitor.html').read_text(encoding='utf-8')
    page=page.replace('开发子集','B2018内层开发').replace('完整2019评价','完整2019评价（本轮不运行）')
    page=page.replace('工程验收样本不代表最终精度。','按内层ES选择EMA，RMSE同步诊断；启动检查不参与选模。')
    page=page.replace('</style><main>','</style><p id="startup_check" style="padding:8px 24px;color:#526a7d"></p><main>')
    page=page.replace('</script></html>', '''</script><script>setInterval(()=>{if(!payload)return;
      const r=payload.runs.find(x=>x.name===selected)||payload.runs.at(-1);
      const s=r?.development.filter(x=>x.startup).at(-1);
      document.getElementById('startup_check').textContent=s?
      `启动检查 ${s.step}步：RMSE ${s.metrics.joint.rmse_mm.toFixed(4)} mm · ES ${s.metrics.joint.value.toFixed(4)} mm（不参与选模）`:'';
    },1000);</script></html>''')
    return page


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default=str(RUN)); p.add_argument('--port',type=int,default=8781)
    p.add_argument('--remote-log',type=Path)
    a=p.parse_args(); page=build_page()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/api/status':
                source=read_remote_status(a.remote_log) if a.remote_log else None
                body=json.dumps(monitor_payload(source or collect_local_status(a.root)),ensure_ascii=False,allow_nan=False).encode()
                kind='application/json; charset=utf-8'
            elif self.path in ('/','/index.html'):
                body=page.encode(); kind='text/html; charset=utf-8'
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def log_message(self,*args):
            pass
    print(f'Grid processor monitor: http://127.0.0.1:{a.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':
    main()
