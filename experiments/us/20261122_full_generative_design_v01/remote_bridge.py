"""Fixed-release transfer, bounded submission and read-only result collection."""
import sys,json,hashlib,os,time,subprocess,tarfile,shlex,base64
from pathlib import Path
BASE=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117'
RELEASE=BASE/'code/20261122_generative_full_v02'
RUN=BASE/'runs/20261122_us_zwd_gan_diffusion_vae_full_a800_v01'
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
    name='20261122_generative_full_v02_code.tar.gz' if header['kind']=='code' else f'20261122_generative_full_{header["kind"]}.tar.gz'
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
    import fcntl
    RELEASE.parent.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    archives={kind:digest(UPLOADS/('20261122_generative_full_v02_code.tar.gz' if kind=='code' else f'20261122_generative_full_{kind}.tar.gz')) for kind in ('data','code')}
    with (RELEASE.parent/'.generative_full_deploy.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if RELEASE.exists():
            if not(RELEASE/'release_complete.json').is_file() or json.loads((RELEASE/'release_complete.json').read_text())!=archives:
                raise ValueError('Immutable release path already belongs to another bundle')
        else:
            staging=RELEASE.parent/('._generative_full_'+archives['code'][:12])
            extract(UPLOADS/'20261122_generative_full_v02_code.tar.gz',staging)
            data_root=BASE/'data'/('generative_full_'+archives['data'][:12])
            if not data_root.exists():
                temporary=data_root.with_name('._'+data_root.name)
                extract(UPLOADS/'20261122_generative_full_data.tar.gz',temporary)
                for name,row in json.loads((temporary/'server_data_manifest.json').read_text())['files'].items():
                    if digest(temporary/name)!=row['sha256']:raise ValueError('Data hash differs:'+name)
                temporary.rename(data_root)
            manifest=json.loads((data_root/'server_data_manifest.json').read_text())
            for name,row in manifest['files'].items():
                if digest(data_root/name)!=row['sha256']:raise ValueError('Frozen input changed:'+name)
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():target.symlink_to(data_root/name)
            (staging/'server_data_manifest.json').write_bytes((data_root/'server_data_manifest.json').read_bytes())
            (staging/'release_complete.json').write_text(json.dumps(archives));staging.rename(RELEASE)
        for manifest_name in ('code_manifest.json','server_data_manifest.json'):
            for name,row in json.loads((RELEASE/manifest_name).read_text())['files'].items():
                if digest(RELEASE/name)!=row['sha256']:raise ValueError('Release SHA differs:'+name)
        env=os.environ.copy();env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',
            PYTHONPATH=str(RELEASE/'src')+':'+str(RELEASE/'tools'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLBACKEND='Agg')
        # Only this project-owned interpreter receives missing plotting wheels.
        checked([str(PY),'-c','import sys;from pathlib import Path;assert Path(sys.prefix).resolve()==(Path.home()/".conda/envs/ZhouSiyu").resolve()'],env)
        wheels_archive=UPLOADS/'20261122_generative_full_environment.tar.gz'
        wheels_root=BASE/'env_packages'/('generative_plot_'+digest(wheels_archive)[:12])
        extract(wheels_archive,wheels_root)
        for name,row in json.loads((wheels_root/'environment_manifest.json').read_text())['files'].items():
            if digest(wheels_root/name)!=row['sha256']:raise ValueError('Offline wheel hash differs:'+name)
        installed=checked([str(PY),'-m','pip','install','--no-index','--no-cache-dir','--find-links',str(wheels_root),
                           'matplotlib==3.9.4'],env)
        (RUN/'environment_install.log').write_text(installed.stdout+installed.stderr)
        checked([str(PY),'-m','pip','check'],env)
        checked([str(PY),'-c','import torch,numpy,scipy,pandas,pyproj,matplotlib;print(torch.__version__,torch.version.cuda)'],env)
        config=RELEASE/'configs/full_generative_a800_v01.json'
        # Planning on the login node is light; the full audit runs in a CPU job.
        checked([str(PY),str(RELEASE/'tools/run_full_generative_campaign.py'),'--config',str(config),'plan','--qualified-hours','26262'],env)
        if (RUN/'submission.json').exists():
            emit(stage='SUBMITTED',result=json.loads((RUN/'submission.json').read_text()),release=str(RELEASE),run=str(RUN));return
        queue=checked(['squeue','-u',os.environ.get('USER','u23114'),'-h','-o','%i|%P|%T|%b']).stdout
        if any('a800' in row.lower() for row in queue.splitlines()):
            raise ValueError('An A800 allocation is already active or queued; single-card submission deferred')
        if (RUN/'submission_intent.json').exists():raise ValueError('Unresolved submission intent; inspect recorded token before retry')
        commands={
            'audit':[str(PY),str(RELEASE/'tools/run_full_generative_campaign.py'),'--config',str(config),'audit'],
            'preflight':[str(PY),str(RELEASE/'tools/run_full_generative_preflight.py'),'--config',str(config),'preflight'],
            'handoff':[str(PY),str(RELEASE/'tools/run_full_generative_preflight.py'),'--config',str(config),'handoff']}
        common=['sbatch','--parsable','--no-requeue','--chdir',str(RELEASE),'--export=ALL']
        gpu=common+['--partition','A800','--gres=gpu:a800:1','--cpus-per-task=2','--mem=48G','--time=60','--signal=B:USR1@180',
            '--job-name','gvd-full-preflight','--output',str(RUN/'preflight-%j.out'),'--error',str(RUN/'preflight-%j.err'),
            '--wrap','exec '+shlex.join(commands['preflight'])]
        checked(gpu[:1]+['--test-only']+gpu[2:],env)
        intent={'created_unix':time.time(),'release_sha256':archives,'max_preflight_card_hours':1,'max_total_card_hours':24}
        (RUN/'submission_intent.json').write_text(json.dumps(intent))
        cpu=common+['--partition','intel','--cpus-per-task=2','--mem=16G','--time=20','--job-name','gvd-full-data-audit',
            '--output',str(RUN/'audit-%j.out'),'--error',str(RUN/'audit-%j.err'),'--wrap','exec '+shlex.join(commands['audit'])]
        audit=checked(cpu,env).stdout.strip().split(';')[0]
        if not audit.isdigit():raise ValueError('Ambiguous CPU sbatch response')
        intent['audit_job_id']=audit;(RUN/'submission_intent.json').write_text(json.dumps(intent))
        preflight=checked(gpu+['--dependency','afterok:'+audit],env).stdout.strip().split(';')[0]
        if not preflight.isdigit():raise ValueError('Ambiguous GPU sbatch response')
        intent['preflight_job_id']=preflight;(RUN/'submission_intent.json').write_text(json.dumps(intent))
        handoff=checked(common+['--partition','intel','--cpus-per-task=2','--mem=4G','--time=10','--job-name','gvd-full-handoff',
            '--dependency','afterany:'+preflight,'--output',str(RUN/'handoff-%j.out'),'--error',str(RUN/'handoff-%j.err'),
            '--wrap','exec '+shlex.join(commands['handoff'])],env).stdout.strip().split(';')[0]
        result=dict(**intent,handoff_job_id=handoff,config=str(config),release=str(RELEASE),run=str(RUN),submitted_unix=time.time())
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
        elif request['action']=='download':download(request)
        else:raise ValueError('Unknown operation')
    except Exception as exc:emit(stage='ERROR',error=str(exc))
