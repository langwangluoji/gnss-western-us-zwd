"""Local-only read-only generation monitor; persists one latest server snapshot."""
import argparse,json,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from grid_pair_monitor import build_page,read_remote_status
from local_context_monitor import collect_local_status
from gnss_field.artifacts import write_json


def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--remote-log',type=Path,required=True); p.add_argument('--port',type=int,default=8783); a=p.parse_args()
 page=build_page().replace('<title>GNSS训练监控</title>','<title>GNSS生成式训练 v02</title>')
 page=page.replace('<h1>GNSS场训练监控</h1>','<h1>GNSS 随机残差 / Diffusion / GAN</h1><p>v02 · 共同均值：冻结M0＋IDW · 累计单A800上限4卡时（含前次短测） · B2018增量开发；C/D未使用</p>')
 page=page.replace('按内层ES选择EMA，RMSE同步诊断；启动检查不参与选模。','点产品按站宏RMSE、联合产品按200km站组宏ES选EMA；启动检查不参与选模。')
 page=page.replace('B2018内层开发联合ES','B2018内层开发站组宏ES')
 class Handler(BaseHTTPRequestHandler):
  def do_GET(self):
   if self.path=='/api/status':
    payload=read_remote_status(a.remote_log) if a.remote_log.exists() else None
    if payload:
     write_json(a.root/'latest_server_snapshot.json',payload)
    else: payload=collect_local_status(a.root)
    for run in payload['runs']: run['status']['stage']='B2018生成式内层开发 · '+run['name']
    body=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8'); kind='application/json; charset=utf-8'
   elif self.path in ('/','/index.html'): body=page.encode('utf-8'); kind='text/html; charset=utf-8'
   else: self.send_error(404); return
   self.send_response(200); self.send_header('Content-Type',kind); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
  def log_message(self,*args): pass
 print(f'Generation monitor http://127.0.0.1:{a.port}',flush=True)
 ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__': main()
