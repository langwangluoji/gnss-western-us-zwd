"""Fixed-release transfer, bounded submission and read-only result collection."""
import sys,json,hashlib,os,time,subprocess,tarfile,shlex,base64
from pathlib import Path
BASE=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117'
RELEASE=BASE/'code/20261122_generative_full_v03'
RUN=BASE/'runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v02'
UPLOADS=BASE/'uploads';PY=Path.home()/'.conda/envs/ZhouSiyu/bin/python'
if not BASE.is_dir():raise ValueError('Authorized project root missing')
for path in (RELEASE,RUN,UPLOADS):
    if not path.resolve().is_relative_to(BASE.resolve()):raise ValueError('Path escapes authorized project')
UPLOADS.mkdir(exist_ok=True)

def emit(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def receive(header):
    if header['kind'] not in ('data','code','environment'):raise ValueError('Unknown bundle')
    name='20261122_generative_full_v03_code.tar.gz' if header['kind']=='code' else f'20261122_generative_full_{header["kind"]}.tar.gz'
    path=UPLOADS/name;part=path.with_suffix('.part')
    if path.exists() and path.stat().st_size==header['bytes'] and digest(path)==header['sha256']:
        emit(stage='OFFSET',offset=header['bytes']);return path
    offset=part.stat().st_size if part.exists() else 0
    if offset>header['bytes']:raise ValueError('Unexpected partial size')
    emit(stage='OFFSET',offset=offset);remaining=header['bytes']-offset
    with part.open('ab') as stream:
        while remaining:
            block=sys.stdin.buffer.read(min(remaining,2**20))
            if not block:raise EOFError('Upload interrupted')
            stream.write(block);remaining-=len(block)
    if digest(part)!=header['sha256']:raise ValueError('Bundle hash differs')
    part.replace(path);return path
def extract(path,directory):
    directory.mkdir(parents=True,exist_ok=True)
    with tarfile.open(path) as tar:
        for member in tar:
            target=(directory/member.name).resolve()
            if not target.is_relative_to(directory.resolve()) or not(member.isfile() or member.isdir()):raise ValueError('Unsafe bundle member')
        tar.extractall(directory,filter='data')
def checked(command,env=None):
    result=subprocess.run(command,capture_output=True,text=True,env=env)
    if result.returncode:raise RuntimeError(str(command[:3])+': '+result.stderr[-5000:]+result.stdout[-2000:])
    return result
def deploy():
    import fcntl,shutil
    from datetime import datetime
    archive=UPLOADS/'20261122_generative_full_v03_code.tar.gz'
    identity=dict(code=digest(archive),revision='runtime_v03',scientific_epochs_unchanged=True)
    RELEASE.parent.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    with (RELEASE.parent/'.generative_full_v03_deploy.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if RELEASE.exists():
            if not(RELEASE/'release_complete.json').exists() or json.loads((RELEASE/'release_complete.json').read_text())!=identity:
                raise ValueError('Runtimev03 release is immutable')
        else:
            staging=RELEASE.parent/('._generative_runtime_'+identity['code'][:12])
            extract(archive,staging)
            for name,row in json.loads((staging/'code_manifest.json').read_text())['files'].items():
                if digest(staging/name)!=row['sha256']:raise ValueError('Runtime source hash differs:'+name)
            (staging/'release_complete.json').write_text(json.dumps(identity));staging.rename(RELEASE)
        for name,row in json.loads((RELEASE/'code_manifest.json').read_text())['files'].items():
            if digest(RELEASE/name)!=row['sha256']:raise ValueError('Runtime release file changed:'+name)
        old=BASE/'runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v01'
        config=RELEASE/'configs/full_generative_a800_v02.json'
        specification=json.loads(config.read_text())
        for name in ('audit_B.json','audit_C.json'):
            value=(old/name).read_bytes();target=RUN/name
            if target.exists() and target.read_bytes()!=value:raise ValueError('Audited population changed:'+name)
            if not target.exists():target.write_bytes(value)
        source=old/'benchmark.json';target=RUN/'original_benchmark.json'
        if target.exists() and digest(target)!=digest(source):raise ValueError('Original measured source changed')
        if not target.exists():shutil.copyfile(source,target)
        caches=((old/'cache',Path(specification['data']['cache_root'])),(old/'benchmark_cache',RUN/'benchmark_cache'))
        for source,target in caches:
            for path in (source,target):
                if not path.resolve().is_relative_to(BASE.resolve()):raise ValueError('Cache path escapes owned project')
            if source.is_dir() and not target.exists():shutil.copytree(source,target)
        env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(RELEASE/'src')+':'+str(RELEASE/'tools'),
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLBACKEND='Agg')
        probe="from gnss_field.generative.campaign import load_config,build_data;from gnss_field.artifacts import read_json,fingerprint;from pathlib import Path;c=load_config("+repr(str(config))+");d=build_data(c);r=Path(c['runtime']['run_root']);b=read_json(r/'original_benchmark.json');assert fingerprint(d.metadata)==b['condition_identity'],'Condition provenance changed';a=read_json(r/'audit_B.json');assert d.identity==a['condition_identity'];assert d._label_identity('B')==a['labels_identity'];print('CONDITION_AND_TRAINING_RELEASE_EXACT')"
        checked([str(PY),'-c',probe],env)
        checked([str(PY),str(RELEASE/'tools/run_full_generative_campaign.py'),'--config',str(config),'plan','--qualified-hours','26262'],env)
        if (RUN/'submission.json').exists():emit(stage='SUBMITTED',result=json.loads((RUN/'submission.json').read_text()));return
        if (RUN/'submission_intent.json').exists():raise ValueError('Unresolved runtime submission intent')
        queue=checked(['squeue','-u',os.environ.get('USER','u23114'),'-h','-o','%i|%P|%T']).stdout
        if any('a800' in row.lower() for row in queue.splitlines()):raise ValueError('Another A800 allocation is active; single-card limit')
        # Previous benchmark3697508 consumed356s. 54min=3240s retains total<=3596s.
        common=['sbatch','--parsable','--no-requeue','--chdir',str(RELEASE),'--export=ALL']
        command=[str(PY),str(RELEASE/'tools/run_full_generative_preflight.py'),'--config',str(config)]
        gpu=common+['--partition','A800','--gres=gpu:a800:1','--cpus-per-task=8','--mem=48G','--time=54','--signal=B:USR1@180',
            '--job-name','gvd-runtime-v03-preflight','--output',str(RUN/'preflight-%j.out'),'--error',str(RUN/'preflight-%j.err'),
            '--wrap','exec '+shlex.join(command+['preflight'])]
        checked(gpu[:1]+['--test-only']+gpu[2:],env)
        intent=dict(created_unix=time.time(),release=identity,previous_card_seconds=356,preflight_total_cap_seconds=3600,max_total_card_hours=24)
        (RUN/'submission_intent.json').write_text(json.dumps(intent))
        job=checked(gpu,env).stdout.strip().split(';')[0]
        if not job.isdigit():raise ValueError('Ambiguous runtime GPU submission')
        intent['preflight_job_id']=job;(RUN/'submission_intent.json').write_text(json.dumps(intent))
        follow=checked(common+['--partition','intel','--cpus-per-task=2','--mem=4G','--time=10','--job-name','gvd-runtime-v03-handoff',
            '--dependency','afterany:'+job,'--output',str(RUN/'handoff-%j.out'),'--error',str(RUN/'handoff-%j.err'),
            '--wrap','exec '+shlex.join(command+['handoff'])],env).stdout.strip().split(';')[0]
        result=dict(**intent,handoff_job_id=follow,config=str(config),run=str(RUN),release_directory=str(RELEASE))
        (RUN/'submission.json').write_text(json.dumps(result));emit(stage='SUBMITTED',result=result)

def status():
    record=dict(stage='SERVER_STATUS',updated_unix=time.time(),run=str(RUN),release=str(RELEASE))
    for name in ('submission.json','preflight_status.json','campaign.json','budget_estimate.json','budget.lock.json','first_training_submission.json','results_export.json'):
        path=RUN/name
        if path.exists() and path.stat().st_size<2**20:record[name]=json.loads(path.read_text())
    record['queue']=checked(['squeue','-u',os.environ.get('USER','u23114'),'-h','-o','%i|%P|%T|%j']).stdout
    ids=[]
    if (RUN/'submission.json').exists():
        sub=json.loads((RUN/'submission.json').read_text());ids=[sub[k] for k in ('audit_job_id','preflight_job_id','handoff_job_id') if sub.get(k)]
    if (RUN/'campaign.json').exists():
        manifest=json.loads((RUN/'campaign.json').read_text());ids.extend(j['job_id'] for j in manifest['jobs'])
    if ids:record['accounting']=checked(['sacct','-X','-j',','.join(sorted(set(ids))),'-n','-P','-o','JobID,State,ElapsedRaw,ExitCode']).stdout
    record['progress']={}
    for path in (RUN/'stages').glob('*/progress.json'):
        record['progress'][path.parent.name]=json.loads(path.read_text())
    record['log_tails']={}
    for path in sorted(list(RUN.glob('*.err'))+list(RUN.glob('*.out')),key=lambda p:p.stat().st_mtime)[-6:]:
        with path.open(errors='replace') as stream:lines=stream.readlines()[-15:]
        record['log_tails'][path.name]=''.join(lines)[-5000:]
    emit(**record)
def download(request):
    path=(RUN/request['relative']).resolve()
    if not path.is_relative_to(RUN.resolve()) or not path.is_file() or path.stat().st_size>512*2**20:
        raise ValueError('Download must name an existing result within this run, <=512MiB')
    emit(stage='DOWNLOAD_START',relative=request['relative'],bytes=path.stat().st_size,sha256=digest(path))
    with path.open('rb') as stream:
        while block:=stream.read(49152):emit(stage='DOWNLOAD_CHUNK',data=base64.b64encode(block).decode('ascii'))
    emit(stage='DOWNLOAD_END',relative=request['relative'])
emit(stage='READY',python=sys.version,home=str(Path.home()))
while line:=sys.stdin.buffer.readline():
    try:
        request=json.loads(line)
        if request['action']=='upload':
            path=receive(request);emit(stage='UPLOADED',kind=request['kind'],bytes=path.stat().st_size)
        elif request['action']=='deploy':deploy()
        elif request['action']=='status':status()
        elif request['action']=='project_script':
            result=subprocess.run([str(PY),'-u','-c',request['source']],cwd=RELEASE,capture_output=True,text=True,timeout=900)
            emit(stage='ACTION_RESULT',exit_code=result.returncode,stdout=result.stdout[-200000:],stderr=result.stderr[-20000:])
        elif request['action']=='download':download(request)
        else:raise ValueError('Unknown operation')
    except Exception as exc:emit(stage='ERROR',error=str(exc))
