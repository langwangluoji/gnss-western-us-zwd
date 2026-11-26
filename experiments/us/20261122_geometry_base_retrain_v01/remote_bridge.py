"""Framed, path-contained transfer and submission over one authenticated SSH channel."""
import sys,json,hashlib,os,time,subprocess,tarfile
from pathlib import Path
BASE=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117'
RELEASE=BASE/'code/20261122_geometry_base_retrain_v01'
RUN=BASE/'runs/20261122_us_dense_zwd_geometry_base_a800_v01'
UPLOADS=BASE/'uploads'
if not BASE.is_dir():raise ValueError('Authorized project root is missing')
for path in (RELEASE,RUN,UPLOADS):
    if not path.resolve().is_relative_to(BASE.resolve()):raise ValueError('Project path escapes authorized root')
UPLOADS.mkdir(parents=True,exist_ok=True)
PY=Path.home()/'.conda/envs/ZhouSiyu/bin/python'


def emit(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(2**20):h.update(block)
    return h.hexdigest()


def read_exact(n,target):
    with target.open('ab') as f:
        while n:
            block=sys.stdin.buffer.read(min(n,2**20))
            if not block:raise EOFError('SSH stream interrupted')
            f.write(block);n-=len(block)


def receive(header):
    if header['kind'] not in ('data','code'):raise ValueError('Unknown upload kind')
    path=UPLOADS/('20261122_geometry_'+header['kind']+'.tar.gz');part=path.with_suffix('.part')
    if path.exists() and path.stat().st_size==header['bytes'] and digest(path)==header['sha256']:
        emit(stage='OFFSET',offset=header['bytes']);return path
    offset=part.stat().st_size if part.exists() else 0
    if offset>header['bytes']:raise ValueError('Partial upload is larger than registered artifact')
    emit(stage='OFFSET',offset=offset);read_exact(header['bytes']-offset,part)
    if digest(part)!=header['sha256']:raise ValueError('Uploaded archive hash mismatch')
    part.replace(path);return path


def extract(path,directory):
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            target=(directory/member.name).resolve()
            if not target.is_relative_to(directory.resolve()) or not(member.isfile() or member.isdir()):raise ValueError('Archive member escapes release')
        tar.extractall(directory,filter='data')


def deploy():
    import fcntl
    for path in (RELEASE,RUN,UPLOADS):
        if not path.resolve().is_relative_to(BASE.resolve()):raise ValueError('Project path resolves outside authorized root')
    archives={kind:digest(UPLOADS/f'20261122_geometry_{kind}.tar.gz') for kind in ('data','code')}
    data_directory=BASE/'data'/('20261122_geometry_frozen_'+archives['data'][:12])
    RELEASE.parent.mkdir(parents=True,exist_ok=True)
    with (RELEASE.parent/'.geometry_deploy.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if RELEASE.exists():
            marker=RELEASE/'release_complete.json'
            if not marker.exists() or json.loads(marker.read_text())!=archives:raise ValueError('Release is immutable; use a new version rather than overwrite existing code')
        else:
            staging=RELEASE.parent/('._geometry_'+archives['code'][:12]);staging.mkdir(exist_ok=True)
            data_directory.parent.mkdir(parents=True,exist_ok=True)
            if data_directory.exists():
                if (data_directory/'archive.sha256').read_text()!=archives['data']:raise ValueError('Frozen data identity mismatch')
            else:
                temporary=data_directory.with_name('._'+data_directory.name);temporary.mkdir(exist_ok=True)
                extract(UPLOADS/'20261122_geometry_data.tar.gz',temporary)
                for name,row in json.loads((temporary/'server_data_manifest.json').read_text())['files'].items():
                    if digest(temporary/name)!=row['sha256']:raise ValueError('Frozen data hash mismatch:'+name)
                (temporary/'archive.sha256').write_text(archives['data']);temporary.rename(data_directory)
            extract(UPLOADS/'20261122_geometry_code.tar.gz',staging)
            data_manifest=json.loads((data_directory/'server_data_manifest.json').read_text())
            for name,row in data_manifest['files'].items():
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() or target.is_symlink():
                    if digest(target)!=row['sha256']:raise ValueError('Code/data file collision:'+name)
                else:target.symlink_to(data_directory/name)
            (staging/'server_data_manifest.json').write_bytes((data_directory/'server_data_manifest.json').read_bytes())
            for manifest in ('server_data_manifest.json','code_manifest.json'):
                for name,row in json.loads((staging/manifest).read_text())['files'].items():
                    expected=row['sha256'] if isinstance(row,dict) else row
                    if digest(staging/name)!=expected:raise ValueError('Staged release hash mismatch:'+name)
            (staging/'release_complete.json').write_text(json.dumps(archives));staging.rename(RELEASE)
    for manifest in ('server_data_manifest.json','code_manifest.json'):
        spec=json.loads((RELEASE/manifest).read_text())
        for name,row in spec['files'].items():
            expected=row['sha256'] if isinstance(row,dict) else row
            if digest(RELEASE/name)!=expected:raise ValueError('Release file hash mismatch:'+name)
    package=RELEASE/'results/20261122_geometry_base_retrain_v01';RUN.mkdir(parents=True,exist_ok=True)
    for name in ('B_cache','PLAN.md','mean.lock.json','probability.lock.json','probability_factors.npy','annual_results.json'):
        target=RUN/name;source=package/name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve()!=source.resolve():raise ValueError('Run input path already belongs to another artifact:'+name)
        else:target.symlink_to(source,target_is_directory=source.is_dir())
    env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONPATH=':'.join(str(RELEASE/p) for p in ('src','tools','results/20261122_geometry_base_retrain_v01')))
    pre=subprocess.run(['sbatch','--test-only','--no-requeue',str(package/'A800.sh')],capture_output=True,text=True,env=env)
    if pre.returncode:raise ValueError('Slurm test-only rejected:'+pre.stderr)
    result=subprocess.run([str(PY),str(package/'server_submit.py'),'--release',str(RELEASE),'--run',str(RUN)],capture_output=True,text=True,env=env)
    if result.returncode:raise ValueError('Submission rejected:'+result.stderr+result.stdout)
    emit(stage='SUBMITTED',result=json.loads(result.stdout),preflight=pre.stderr[-2000:],release=str(RELEASE),run=str(RUN))


def status():
    record=dict(stage='SERVER_STATUS',updated=time.time(),run=str(RUN))
    for name in ('status.json','submission.json','server_benchmark.json','server_engineering.json'):
        if (RUN/name).exists():record[name]=json.loads((RUN/name).read_text())
    q=subprocess.run(['squeue','-u',os.environ['USER'],'-h','-o','%i|%P|%T|%j'],capture_output=True,text=True);record['queue']=q.stdout
    if (RUN/'job_id').exists():
        jid=(RUN/'job_id').read_text().strip();record['job_id']=jid
        record['accounting']=subprocess.run(['sacct','-X','-j',jid,'-n','-P','-o','JobID,State,ElapsedRaw,ExitCode'],capture_output=True,text=True).stdout
        for suffix in ('out','err'):
            path=RUN/'logs'/f'slurm_{jid}.{suffix}'
            if path.exists():record['log_'+suffix]='\n'.join(path.read_text(errors='replace').splitlines()[-25:])
    emit(**record)


emit(stage='READY',home=str(Path.home()),python=sys.version)
while line:=sys.stdin.buffer.readline():
    try:
        request=json.loads(line)
        if request['action']=='upload':
            path=receive(request);emit(stage='UPLOADED',kind=request['kind'],bytes=path.stat().st_size)
        elif request['action']=='deploy':deploy()
        elif request['action']=='status':status()
        else:raise ValueError('Unknown bridge operation')
    except Exception as exc:emit(stage='ERROR',error=str(exc))
