"""Interactive SSH authentication; credentials are handled only by OpenSSH's console."""
import sys,subprocess,json,time,queue,threading
from pathlib import Path
from retrain_common import ROOT

BRIDGE=ROOT/'bridge';BRIDGE.mkdir(exist_ok=True)
config=json.loads((BRIDGE/'config.json').read_text());transport=config['transport']
bootstrap="import sys;exec(sys.stdin.buffer.read(int(sys.stdin.buffer.readline())))"
remote="$HOME/.conda/envs/ZhouSiyu/bin/python -u -c "+__import__('shlex').quote(bootstrap)
args=[str(Path(__import__('os').environ['SystemRoot'])/'System32/OpenSSH/ssh.exe'),'-T','-p','10022','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=60','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','-o',
      f'ProxyCommand=D:/anaconda3-2025.12-2/envs/pytorch/python.exe {transport} logini.tongji.edu.cn 10022','u23114@logini.tongji.edu.cn',remote]
process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None)
(BRIDGE/'ssh_pid').write_text(str(process.pid));messages=queue.Queue()


def receive():
    for raw in iter(process.stdout.readline,b''):
        line=raw.decode('utf-8','replace').strip()
        with (BRIDGE/'live.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
        try:
            message=json.loads(line);messages.put(message)
            if message.get('stage')=='SERVER_STATUS':
                tmp=ROOT/'server_status.tmp';tmp.write_text(json.dumps(message,ensure_ascii=False),encoding='utf-8');tmp.replace(ROOT/'server_status.json')
        except json.JSONDecodeError:print(line,flush=True)
    messages.put(dict(stage='DISCONNECTED'))


threading.Thread(target=receive,daemon=True).start()
payload=(ROOT/'remote_bridge.py').read_bytes();process.stdin.write(str(len(payload)).encode()+b'\n'+payload);process.stdin.flush()


def wait(stage):
    while True:
        m=messages.get()
        if m.get('stage') in ('ERROR','DISCONNECTED'):raise RuntimeError(str(m))
        if m.get('stage')==stage:return m


def send(message):process.stdin.write(json.dumps(message).encode()+b'\n');process.stdin.flush()


def upload(kind):
    meta_path=ROOT/('server_data_bundle.json' if kind=='data' else 'server_code_bundle.json')
    while not meta_path.exists():
        if process.poll() is not None:raise RuntimeError('SSH ended while waiting for verified code')
        print('Waiting for verified code artifact; no GPU submitted.',flush=True);time.sleep(10)
    meta=json.loads(meta_path.read_text());path=ROOT/('server_data.tar.gz' if kind=='data' else 'server_code.tar.gz')
    send(dict(action='upload',kind=kind,bytes=meta['bytes'],sha256=meta['sha256']));offset=wait('OFFSET')['offset'];last=0
    with path.open('rb') as f:
        f.seek(offset)
        while block:=f.read(2**20):
            process.stdin.write(block);offset+=len(block)
            if time.monotonic()-last>5:
                record=dict(stage='UPLOADING',kind=kind,bytes_sent=offset,total=meta['bytes'],updated=time.time());(ROOT/'transfer_status.json').write_text(json.dumps(record),encoding='utf-8');print(record,flush=True);last=time.monotonic()
    process.stdin.flush();print(wait('UPLOADED'),flush=True)


try:
    print(wait('READY'),flush=True);upload('data');upload('code');send(dict(action='deploy'));print(wait('SUBMITTED'),flush=True)
    while process.poll() is None:
        send(dict(action='status'));m=wait('SERVER_STATUS');stage=m.get('status.json',{}).get('stage');print('JOB',m.get('job_id'),stage,m.get('queue','').strip(),flush=True)
        terminal=stage in ('COMPLETE','STOP_BUDGET') or any(s in m.get('accounting','') for s in ('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY'))
        if terminal:
            print('Job terminal; authenticated connection remains idle. Touch bridge/request_status for a fresh check.',flush=True)
            while not (BRIDGE/'request_status').exists() and process.poll() is None:time.sleep(2)
            if (BRIDGE/'request_status').exists():(BRIDGE/'request_status').unlink()
        else:time.sleep(20)
except Exception as exc:
    (ROOT/'bridge_error.json').write_text(json.dumps(dict(error=str(exc),updated=time.time())),encoding='utf-8');print(exc,flush=True)
    raise
