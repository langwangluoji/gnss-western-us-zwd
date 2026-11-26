"""Create authenticated deployment, bounded A800 submission and read-only snapshots."""
from pathlib import Path
import base64,shutil
from gnss_field.artifacts import read_json,sha256,write_json

PROJECT=Path(__file__).resolve().parents[1]
NAME='20261121_grid_generation_v02'


def main():
    record=read_json(PROJECT/'results'/NAME/'release.json'); archive=Path(record['archive'])
    if sha256(archive)!=record['archive_sha256']: raise ValueError('Archive changed')
    payload=base64.b64encode(archive.read_bytes()).decode('ascii')
    folder=PROJECT/'dist'/f'{NAME}_submission'; folder.mkdir(exist_ok=True)
    script='''set -euo pipefail
project_root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
export GRID_RELEASE="$project_root/code/20261121_grid_generation_v02"
export GRID_RUN="$project_root/runs/20261121_us_dense_zwd_gen3_unet_df_gan_v02"
export GRID_PYTHON="$HOME/.conda/envs/ZhouSiyu/bin/python"
export PYTHONNOUSERSITE=1
"$GRID_PYTHON" - <<'PY_DEPLOY'
import base64,shutil,hashlib,json,os,zipfile
from pathlib import Path
root=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117'
release=Path(os.environ['GRID_RELEASE']); run=Path(os.environ['GRID_RUN'])
def digest(p):
 with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
raw=base64.b64decode('''+repr(payload)+''')
assert hashlib.sha256(raw).hexdigest()=='''+repr(record['archive_sha256'])+'''
archive=root/'uploads/20261121_grid_generation_v02_code.zip'
if archive.exists(): assert digest(archive)==hashlib.sha256(raw).hexdigest()
else:
 tmp=archive.with_suffix('.partial'); tmp.write_bytes(raw); tmp.replace(archive)
if not release.exists():
 stage=release.with_name(release.name+'.staging'); assert not stage.exists(),'Existing staging preserved'
 stage.mkdir()
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for p in z.infolist(): assert (stage/p.filename).resolve().is_relative_to(stage.resolve())
  z.extractall(stage)
 assert digest(stage/'grid_release.json')=='''+repr(record['release_sha256'])+'''
 for name,value in json.loads((stage/'grid_release.json').read_text())['files'].items():
  assert (stage/name).resolve().is_relative_to(stage.resolve()) and digest(stage/name)==value,name
 stage.rename(release)
assert digest(release/'grid_release.json')=='''+repr(record['release_sha256'])+'''
old=root/'runs/20261121_us_dense_zwd_grid69_processor_pair_v01'
assert json.loads((old/'queue_status.json').read_text())['status']=='COMPLETE'
for fold in (0,1):
 old_result=json.loads((old/f'fold_{fold}/resunet69/results.json').read_text())
 expected=json.loads((release/f'assets/mean_fold{fold}_source.json').read_text())
 assert old_result==expected,'Old selected mean changed since download'
 assert digest(Path(old_result['selected']['checkpoint']))==old_result['source_model']
run.mkdir(parents=True,exist_ok=True); (run/'logs').mkdir(exist_ok=True)
# Carry every earlier allocation forward; this is not a fresh four-hour grant.
prior=root/'runs/20261121_us_dense_zwd_gen3_unet_df_gan_v01'
lock=json.loads((prior/'allocation.lock.json').read_text())
assert lock['cap_seconds']==14400 and lock['gpus']==1
assert lock['release_sha256']=='61dda56f0fbc85ba6b8b69e3d8063b0ed92d4eb85d8095da56af99246d903651'
history=json.loads((prior/'allocations.json').read_text())
assert any(v['job_id']=='3644321' for v in history)
p=run/'allocations.json'
if not p.exists(): p.write_text(json.dumps(history))
else:
 current=json.loads(p.read_text()); assert all(v in current for v in history)
lineage=run/'lineage.json'
if not lineage.exists(): lineage.write_text(json.dumps(dict(source=str(prior),cap_seconds=14400,carried_allocations=history)))
runtime=dict(source_root=str(old),source_runtime=str(old/'runtime.json'),mean_mode='frozen_m0_idw')
p=run/'runtime.json'
if p.exists(): assert json.loads(p.read_text())==runtime
else: p.write_text(json.dumps(runtime,sort_keys=True))
print('GENERATION_RELEASE_AND_MEANS_VERIFIED',str(release),flush=True)
PY_DEPLOY
export PYTHONPATH="$GRID_RELEASE/src"
"$GRID_PYTHON" "$GRID_RELEASE/tools/grid_pair_release.py" --verify "$GRID_RELEASE"
bash -n "$GRID_RELEASE/slurm/grid_generation_a800.sh"
sbatch --test-only "$GRID_RELEASE/slurm/grid_generation_a800.sh"
"$GRID_PYTHON" "$GRID_RELEASE/tools/grid_generation_submit.py" --root "$GRID_RUN" --release "$GRID_RELEASE" --python "$GRID_PYTHON" --resume
"$GRID_PYTHON" -u - <<'PY_MONITOR'
from pathlib import Path
import os,json,time,subprocess,sys
sys.path.insert(0,str(Path(os.environ['GRID_RELEASE'])/'tools'))
from grid_pair_monitor import stage_status
r=Path(os.environ['GRID_RUN']); jid=(r/'job_id').read_text().strip()
def lines(p):
 try: return [json.loads(x) for x in p.read_text().splitlines()[-100:] if x.strip()]
 except FileNotFoundError: return []
for iteration in range(4320):
 sq=subprocess.run(['squeue','-h','-j',jid,'-o','%T|%R'],capture_output=True,text=True,timeout=15)
 state=sq.stdout.strip().split('|')[0]
 if not state:
  ac=subprocess.run(['sacct','-X','-n','-P','-j',jid,'-o','State'],capture_output=True,text=True,timeout=15)
  state=ac.stdout.strip().split('|')[0].strip() or 'UNKNOWN'
 queue=json.loads((r/'queue_status.json').read_text()) if (r/'queue_status.json').exists() else {'status':'STARTING'}
 runs=[]
 for log in sorted(r.glob('fold_*/*/training.jsonl')):
  tr=lines(log); dev=lines(log.parent/'development.jsonl'); last=tr[-1] if tr else {}
  eng=json.loads((log.parent/'status.json').read_text()) if (log.parent/'status.json').exists() else {}
  products={}
  initial=json.loads((log.parent/'initial_metrics.json').read_text()) if (log.parent/'initial_metrics.json').exists() else {}
  for product in ('point','joint','marginal'):
   candidates=[dict(initial[product],step=0)] if product in initial else []
   candidates += [dict(x['metrics'][product],step=x['step']) for x in dev if not x['startup'] and product in x['metrics']]
   products[product]=sorted(candidates,key=lambda x:(x['value'],x['step']))[:1]
  if queue['status']=='PAUSED_DURING_EVALUATION': eng['status']='PAUSED_DURING_EVALUATION'
  status=dict(stage='B2018 generation pilot',status=stage_status(eng.get('status',''),(log.parent/'results.json').exists(),state),
   step=last.get('step',0),updated=last.get('updated'),products=products)
  runs.append(dict(name=log.parent.relative_to(r).as_posix(),status=status,training=tr,development=dev))
 if state in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PENDING'): queue['status']=state
 print('GRID_SNAPSHOT '+json.dumps(dict(root=str(r),generated_at=time.time(),runs=runs,full2019=[],read_only=True,
  campaign_queue=dict(campaign_id=r.name,state=queue['status'],jobs=[dict(job_id=jid,state=sq.stdout.strip() or state)]),
  last_update=max((x['status']['updated'] or 0 for x in runs),default=None)),allow_nan=False),flush=True)
 if queue['status']=='COMPLETE' or not sq.stdout.strip():
  print('SLURM_FINAL',subprocess.run(['sacct','-j',jid,'-o','JobID,State,ExitCode,Elapsed,AllocTRES'],capture_output=True,text=True).stdout,flush=True)
  for p in sorted((r/'logs').glob('slurm_*.out')): print('LAST_JOB_LOG',p.read_text()[-10000:],flush=True)
  break
 time.sleep(5)
PY_MONITOR
'''
    shutil.copy2(PROJECT/'tools/ssh_tcp_transport.py',folder/'ssh_tcp_transport.py')
    (folder/'submit.stdin').write_text(script,encoding='utf-8',newline='\n')
    (folder/'submit.cmd').write_text('@echo off\r\ntitle GNSS stochastic Diffusion GAN A800\r\n"%SystemRoot%\\System32\\OpenSSH\\ssh.exe" -T -o "ProxyCommand=D:/anaconda3-2025.12-2/envs/pytorch/python.exe %~dp0ssh_tcp_transport.py logini.tongji.edu.cn 10022" -v -E "%~dp0ssh_debug.log" -o ConnectTimeout=120 -o ConnectionAttempts=3 -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 -p 10022 u23114@logini.tongji.edu.cn "bash -s" < "%~dp0submit.stdin" > "%~dp0submit.log"\r\necho SSH exited %ERRORLEVEL%\r\npause\r\n',encoding='ascii',newline='')
    write_json(folder/'provenance.json',dict(**record,credentials_stored=False,training_started=False,gpus=1,allocation_cap_seconds=14400))
    print('GENERATION_SUBMISSION_READY',folder)


if __name__=='__main__': main()
