"""One visible OpenSSH authentication; keep read-only status/download available."""
from pathlib import Path
import base64,hashlib,json,os,queue,shlex,subprocess,threading,time

ROOT=Path(__file__).resolve().parent
BRIDGE=ROOT/'bridge';BRIDGE.mkdir(exist_ok=True)
TRANSPORT='C:/Users/user/AppData/Local/Temp/gnss_geometry_base_a800_v01/ssh_tcp_transport.py'
PY='D:/anaconda3-2025.12-2/envs/pytorch/python.exe'
bootstrap='import sys;exec(sys.stdin.buffer.read(int(sys.stdin.buffer.readline())))'
remote='$HOME/.conda/envs/ZhouSiyu/bin/python -u -c '+shlex.quote(bootstrap)
arguments=[str(Path(os.environ['SystemRoot'])/'System32/OpenSSH/ssh.exe'),'-T','-p','10022',
    '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=60','-o','ServerAliveInterval=30',
    '-o','ServerAliveCountMax=6','-o',f'ProxyCommand={PY} {TRANSPORT} logini.tongji.edu.cn 10022',
    'u23114@logini.tongji.edu.cn',remote]
print('GNSS full VAE GAN Diffusion: enter the Tongji password in this window only.',flush=True)
process=subprocess.Popen(arguments,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None)
(BRIDGE/'ssh_pid').write_text(str(process.pid))
messages=queue.Queue();current_download={}

def receive():
    for raw in iter(process.stdout.readline,b''):
        line=raw.decode('utf-8','replace').strip()
        try:message=json.loads(line)
        except json.JSONDecodeError:
            print(line,flush=True);continue
        stage=message.get('stage')
        if stage=='DOWNLOAD_CHUNK':
            current_download['stream'].write(base64.b64decode(message['data']));continue
        if stage=='DOWNLOAD_START':
            relative=Path(message['relative'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe downloaded relative path')
            target=ROOT/'server_downloads'/message['sha256'][:12]/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            current_download.update(stream=target.with_suffix(target.suffix+'.part').open('wb'),target=target,meta=message)
            continue
        if stage=='DOWNLOAD_END':
            current_download['stream'].close();target=current_download['target'];part=target.with_suffix(target.suffix+'.part')
            with part.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=current_download['meta']['sha256']:raise ValueError('Downloaded hash differs')
            part.replace(target);print('DOWNLOADED',str(target),flush=True);current_download.clear();messages.put(message);continue
        with (BRIDGE/'live.log').open('a',encoding='utf-8') as stream:stream.write(json.dumps(message,ensure_ascii=False)+'\n')
        if stage=='SERVER_STATUS':
            temporary=ROOT/'server_status.tmp';temporary.write_text(json.dumps(message,ensure_ascii=False),encoding='utf-8')
            temporary.replace(ROOT/'server_status.json')
        messages.put(message)
    messages.put(dict(stage='DISCONNECTED'))

threading.Thread(target=receive,daemon=True).start()
payload=(ROOT/'remote_bridge.py').read_bytes()
process.stdin.write(str(len(payload)).encode()+b'\n'+payload);process.stdin.flush()

def wait(stage):
    while True:
        value=messages.get()
        if value.get('stage') in ('ERROR','DISCONNECTED'):raise RuntimeError(str(value))
        if value.get('stage')==stage:return value
def send(value):process.stdin.write(json.dumps(value).encode()+b'\n');process.stdin.flush()
def upload(kind):
    meta=json.loads((ROOT/f'server_{kind}_bundle.json').read_text(encoding='utf-8'))
    path=ROOT/f'server_{kind}.tar.gz'
    send(dict(action='upload',kind=kind,bytes=meta['bytes'],sha256=meta['sha256']))
    offset=wait('OFFSET')['offset'];last=0
    with path.open('rb') as stream:
        stream.seek(offset)
        while block:=stream.read(2**20):
            process.stdin.write(block);offset+=len(block)
            if time.monotonic()-last>5:
                value=dict(kind=kind,bytes_sent=offset,total=meta['bytes'],updated_unix=time.time())
                (ROOT/'transfer_status.json').write_text(json.dumps(value),encoding='utf-8')
                print('UPLOADING',kind,offset,'/',meta['bytes'],flush=True);last=time.monotonic()
    process.stdin.flush();print(wait('UPLOADED'),flush=True)
try:
    print(wait('READY'),flush=True)
    upload('data');upload('code');upload('environment');send(dict(action='deploy'))
    try:print(wait('SUBMITTED'),flush=True)
    except RuntimeError as exc:
        # Keep the authenticated bridge after a recoverable deployment failure.
        print('DEPLOYMENT_NEEDS_REPAIR',str(exc),flush=True)
        (ROOT/'bridge_error.json').write_text(json.dumps(dict(error=str(exc),updated_unix=time.time())),encoding='utf-8')
    last=0
    while process.poll() is None:
        if (BRIDGE/'retry_deploy').exists():
            (BRIDGE/'retry_deploy').unlink()
            send(dict(action='deploy'))
            try:print(wait('SUBMITTED'),flush=True)
            except RuntimeError as exc:print('DEPLOYMENT_NEEDS_REPAIR',str(exc),flush=True)
        download=BRIDGE/'request_download.json'
        if download.exists():
            request=json.loads(download.read_text(encoding='utf-8'));download.unlink()
            send(dict(action='download',relative=request['relative']));wait('DOWNLOAD_END')
        if time.monotonic()-last>=30 or (BRIDGE/'request_status').exists():
            (BRIDGE/'request_status').unlink(missing_ok=True)
            send(dict(action='status'));value=wait('SERVER_STATUS')
            print('SERVER_STATUS',value.get('preflight_status.json',{}).get('state'),value.get('queue','').strip(),flush=True)
            last=time.monotonic()
        time.sleep(2)
except Exception as exc:
    (ROOT/'bridge_error.json').write_text(json.dumps(dict(error=str(exc),updated_unix=time.time())),encoding='utf-8')
    print('BRIDGE_ERROR',str(exc),flush=True);raise
