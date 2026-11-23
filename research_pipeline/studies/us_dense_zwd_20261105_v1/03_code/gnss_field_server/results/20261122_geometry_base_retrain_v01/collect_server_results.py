"""Read-only SSH result collection; OpenSSH alone handles the console password."""
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib,io,json,subprocess,sys,tarfile,time
from pathlib import Path
base=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117'
root=base/'runs/20261122_us_dense_zwd_geometry_base_a800_v01'
if not root.resolve().is_relative_to(base.resolve()):raise ValueError('Unexpected run root')
files={}
def add(path):
    if path.is_file():
        if not path.resolve().is_relative_to(base.resolve()):raise ValueError('Result leaves project')
        files[path.relative_to(root).as_posix()]=path.read_bytes()
for path in root.glob('*.json'):add(path)
for path in (root/'logs').glob('slurm_3688243.*'):add(path)
for fold in (0,1):
    for arm in ('gaussian','gan','diffusion','ndp_point','ndp_GP_point'):
        folder=root/f'fold_{fold}'/arm
        for pattern in ('*.json','*.jsonl','*predictions.npz'):
            for path in folder.glob(pattern):add(path)
        result=folder/'result.json'
        if result.exists():
            for product in json.loads(result.read_text()).get('products',{}).values():
                name=product['selected']['checkpoint']
                if Path(name).name!=name:raise ValueError('Unexpected checkpoint name')
                add(folder/name)
        elif arm=='ndp_GP_point':add(folder/'latest.pt')
metadata={'collected':time.time(),'run':str(root),'job_id':'3688243',
          'accounting':subprocess.run(['sacct','-X','-j','3688243','-n','-P','-o','JobID,State,ElapsedRaw,ExitCode'],capture_output=True,text=True,check=True).stdout,
          'files':{name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for name,data in files.items()}}
files['collection_manifest.json']=json.dumps(metadata,sort_keys=True).encode()
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
    for name,data in sorted(files.items()):
        item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
data=buffer.getvalue()
print(json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'files':len(files)}),flush=True)
sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
'''


def main():
    transport = json.loads((ROOT / 'bridge/config.json').read_text())['transport']
    bootstrap = 'import sys;exec(sys.stdin.buffer.read(int(sys.stdin.buffer.readline())))'
    command = "$HOME/.conda/envs/ZhouSiyu/bin/python -u -c " + shlex.quote(bootstrap)
    args = [str(Path(os.environ['SystemRoot']) / 'System32/OpenSSH/ssh.exe'),
            '-T', '-p', '10022', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=60', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=6', '-o',
            f'ProxyCommand=D:/anaconda3-2025.12-2/envs/pytorch/python.exe {transport} logini.tongji.edu.cn 10022',
            'u23114@logini.tongji.edu.cn', command]
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    (ROOT / 'bridge/collector_pid').write_text(str(process.pid))
    payload = REMOTE.encode()
    process.stdin.write(str(len(payload)).encode() + b'\n' + payload)
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f'SSH ended before result header: {process.wait()}')
    header = json.loads(line)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    archive = ROOT / f'server_results_{stamp}.tar.gz'
    remaining = int(header['bytes'])
    if not 0 < remaining < 512 * 2**20:
        raise ValueError('Unexpected result bundle size')
    sha = hashlib.sha256()
    print('Downloading', header, flush=True)
    with archive.with_suffix('.part').open('wb') as stream:
        while remaining:
            block = process.stdout.read(min(remaining, 2**20))
            if not block:
                raise EOFError('Interrupted result download')
            stream.write(block);sha.update(block);remaining -= len(block)
    if process.wait() or sha.hexdigest() != header['sha256']:
        raise ValueError('SSH exit or bundle hash check failed')
    archive.with_suffix('.part').replace(archive)
    output = ROOT / f'received_{stamp}'
    if os.name == 'nt':
        output = Path('\\\\?\\' + str(output))
    output.mkdir()
    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            if not item.isfile() or not (output / item.name).resolve().is_relative_to(output.resolve()):
                raise ValueError('Unsafe result member')
        tar.extractall(output, filter='data')
    manifest = json.loads((output / 'collection_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = output / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise ValueError('Result file hash mismatch: ' + name)
    record = dict(passed=True, directory=str(output), archive=str(archive), **header)
    (ROOT / 'collection_complete.json').write_text(json.dumps(record), encoding='utf-8')
    print('RESULTS_VERIFIED', json.dumps(record), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        (ROOT / 'collection_error.json').write_text(json.dumps({'error':str(error)}), encoding='utf-8')
        raise
