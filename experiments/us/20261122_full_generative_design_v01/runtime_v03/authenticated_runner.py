"""Ephemeral in-memory SSH authentication; immutable project deployment only."""
from pathlib import Path
import sys,os,getpass,json,shlex,time,hashlib,base64,subprocess,socket
sys.path.insert(0,'C:/Users/user/AppData/Local/Temp/gnss_full_generative_v01/ssh_lib')
import paramiko

ROOT=Path(__file__).resolve().parent;BRIDGE=ROOT/'bridge';BRIDGE.mkdir(exist_ok=True)
client=paramiko.SSHClient()
client.load_system_host_keys(str(Path.home()/'.ssh/known_hosts'))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
secret=getpass.getpass('Tongji SSH password (not stored): ')
connection=socket.create_connection(('logini.tongji.edu.cn',10022),timeout=8)
connection.settimeout(None)
client.connect('logini.tongji.edu.cn',port=10022,username='u23114',password=secret,sock=connection,
               allow_agent=False,look_for_keys=False,timeout=30,banner_timeout=30,auth_timeout=30)
secret=None;client.get_transport().set_keepalive(30)
print('AUTHENTICATED_HOST_KEY_VERIFIED',flush=True)
(BRIDGE/'runner_pid').write_text(str(os.getpid()))
bootstrap='import sys;exec(sys.stdin.buffer.read(int(sys.stdin.buffer.readline())))'
command='$HOME/.conda/envs/ZhouSiyu/bin/python -u -c '+shlex.quote(bootstrap)
stdin,stdout,stderr=client.exec_command(command)
payload=(ROOT/'remote_bridge.py').read_bytes();stdin.write(str(len(payload)).encode()+b'\n'+payload);stdin.flush()

def receive():
    line=stdout.readline()
    if not line:
        raise RuntimeError('SSH channel ended: '+stderr.read().decode('utf8','replace')[-3000:])
    value=json.loads(line)
    with (BRIDGE/'live.log').open('a',encoding='utf8') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')
    return value
def wait(stage):
    while True:
        value=receive()
        if value.get('stage')=='ERROR':raise RuntimeError(value['error'])
        if value.get('stage')==stage:return value
def send(value):stdin.write(json.dumps(value).encode()+b'\n');stdin.flush()
def upload(kind):
    spec=json.loads((ROOT/'bundle.json').read_text(encoding='utf8'))
    send(dict(action='upload',kind=kind,bytes=spec['bytes'],sha256=spec['sha256']))
    offset=wait('OFFSET')['offset']
    with (ROOT/'code.tar.gz').open('rb') as stream:
        stream.seek(offset)
        while block:=stream.read(2**20):stdin.write(block)
    stdin.flush();print('VERIFIED_UPLOAD',wait('UPLOADED'),flush=True)

try:
    print(wait('READY'),flush=True)
    upload('code')
    send(dict(action='deploy'))
    try:
        submitted=wait('SUBMITTED');print('SUBMITTED',json.dumps(submitted),flush=True)
        (ROOT/'submission_verified.json').write_text(json.dumps(submitted,ensure_ascii=False),encoding='utf8')
    except RuntimeError as exc:
        print('DEPLOYMENT_ERROR',str(exc),flush=True)
        (ROOT/'deployment_error.json').write_text(json.dumps(dict(error=str(exc),updated_unix=time.time())),encoding='utf8')
    last=0
    while client.get_transport().is_active():
        action=BRIDGE/'request_action.json'
        if action.exists():
            request=json.loads(action.read_text(encoding='utf8'));action.unlink()
            script=(ROOT/request['script']).resolve()
            if not script.is_relative_to(ROOT.resolve()) or script.suffix!='.py':raise ValueError('Invalid project action script')
            send(dict(action='project_script',source=script.read_text(encoding='utf8')))
            response=wait('ACTION_RESULT')
            (BRIDGE/'action_result.json').write_text(json.dumps(response,ensure_ascii=False),encoding='utf8')
            print('PROJECT_ACTION_RESULT',response,flush=True)
        if (BRIDGE/'request_download.json').exists():
            request=json.loads((BRIDGE/'request_download.json').read_text(encoding='utf8'));(BRIDGE/'request_download.json').unlink()
            send(dict(action='download',relative=request['relative']));header=wait('DOWNLOAD_START')
            relative=Path(header['relative'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe result path')
            target=ROOT/'server_downloads'/header['sha256'][:12]/relative;target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix(target.suffix+'.part')
            with temporary.open('wb') as stream:
                while True:
                    message=receive()
                    if message['stage']=='DOWNLOAD_END':break
                    if message['stage']!='DOWNLOAD_CHUNK':raise ValueError('Unexpected download message')
                    stream.write(base64.b64decode(message['data']))
            with temporary.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=header['sha256']:raise ValueError('Downloaded result hash differs')
            temporary.replace(target);print('RESULT_DOWNLOADED',str(target),flush=True)
        if time.monotonic()-last>15 or (BRIDGE/'request_status').exists():
            (BRIDGE/'request_status').unlink(missing_ok=True);send(dict(action='status'));value=wait('SERVER_STATUS')
            temporary=ROOT/'server_status.tmp';temporary.write_text(json.dumps(value,ensure_ascii=False),encoding='utf8')
            temporary.replace(ROOT/'server_status.json')
            print('STATUS',value.get('preflight_status.json',{}).get('state'),value.get('queue','').strip(),flush=True)
            last=time.monotonic()
        time.sleep(2)
except Exception as exc:
    (ROOT/'runner_error.json').write_text(json.dumps(dict(error=str(exc),updated_unix=time.time())),encoding='utf8')
    raise
finally:
    stdin.close();client.close()
