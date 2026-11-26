"""Prepare a visible SSH login and one bounded Slurm queue; no credentials stored."""
import base64
from pathlib import Path
from gnss_field.artifacts import read_json,sha256,write_json


def main():
    project=Path(__file__).resolve().parents[1]
    record=read_json(project/'results/20261121_grid_processor_pair_v01/release.json')
    archive=Path(record['archive'])
    if sha256(archive)!=record['archive_sha256']:
        raise ValueError('Release archive changed')
    payload=base64.b64encode(archive.read_bytes()).decode('ascii')
    folder=project/'dist/20261121_grid_processor_pair_v01_submission'
    folder.mkdir(exist_ok=True)
    script='''set -euo pipefail
project_root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
export GRID_RELEASE="$project_root/code/20261121_grid_processor_pair_v01"
export GRID_RUN="$project_root/runs/20261121_us_dense_zwd_grid69_processor_pair_v01"
export GRID_PYTHON="$HOME/.conda/envs/ZhouSiyu/bin/python"
export PYTHONNOUSERSITE=1
"$GRID_PYTHON" - <<'PY_DEPLOY'
import base64,hashlib,json,os,shutil,zipfile
from pathlib import Path
root=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117'
release=Path(os.environ['GRID_RELEASE']); run=Path(os.environ['GRID_RUN'])
expected='''+repr(record['release_sha256'])+'''
bundle_hashes='''+repr(record['reused_bundle'])+'''
def digest(path):
 with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
data=base64.b64decode('''+repr(payload)+''')
assert hashlib.sha256(data).hexdigest()=='''+repr(record['archive_sha256'])+'''
archive=root/'uploads/20261121_grid_processor_pair_v01_code.zip'
if archive.exists(): assert digest(archive)==hashlib.sha256(data).hexdigest()
else:
 temporary=archive.with_suffix('.partial'); temporary.write_bytes(data); temporary.replace(archive)
if not release.exists():
 stage=release.with_name(release.name+'.staging')
 assert not stage.exists(), 'Existing staging preserved; inspect before retry'
 stage.mkdir()
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for member in z.infolist(): assert (stage/member.filename).resolve().is_relative_to(stage.resolve())
  z.extractall(stage)
 # Existing immutable bundle is reused only if every registered byte matches.
 candidates=[p.parent for p in (root/'code').glob('*/bundle/data.npz')]
 source=next((p for p in candidates if all((p/k).is_file() and digest(p/k)==v for k,v in bundle_hashes.items())),None)
 assert source is not None, 'No checksum-matched existing B2018 bundle; upload required before any job'
 (stage/'bundle').mkdir()
 for name in bundle_hashes: shutil.copy2(source/name,stage/'bundle'/name)
 manifest=stage/'grid_release.json'; assert digest(manifest)==expected
 for name,value in json.loads(manifest.read_text())['files'].items():
  p=(stage/name).resolve(); assert p.is_relative_to(stage.resolve()) and digest(p)==value,name
 stage.rename(release)
assert digest(release/'grid_release.json')==expected
for name,value in json.loads((release/'grid_release.json').read_text())['files'].items():
 p=(release/name).resolve(); assert p.is_relative_to(release.resolve()) and digest(p)==value,name
run.mkdir(parents=True,exist_ok=True); (run/'logs').mkdir(exist_ok=True)
if not (run/'cache/features69.npy').exists():
 assert shutil.disk_usage(run).free>40*2**30, 'At least 40 GiB project free space required for FP32 cache'
runtime=dict(data_root=str(root/'data/gnss_field_server_v1'),m0=str(release/'assets/M0_005000.pt'),
 bundle=str(release/'bundle'),design=str(release/'assets/design.json'),outer=str(release/'assets/outer.json'),
 parameters=str(release/'assets/mixture.json'))
runtime_path=run/'runtime.json'
if runtime_path.exists(): assert json.loads(runtime_path.read_text())==runtime
else: runtime_path.write_text(json.dumps(runtime,sort_keys=True))
print('GRID_RELEASE_AND_FROZEN_ASSETS_VERIFIED',str(release),flush=True)
PY_DEPLOY
export PYTHONPATH="$GRID_RELEASE/src"
"$GRID_PYTHON" "$GRID_RELEASE/tools/grid_pair_release.py" --verify "$GRID_RELEASE"
squeue -u "$USER" -h -o '%i|%j|%P|%T|%R'
bash -n "$GRID_RELEASE/slurm/grid_pair_a800.sh"
sbatch --test-only "$GRID_RELEASE/slurm/grid_pair_a800.sh"
"$GRID_PYTHON" "$GRID_RELEASE/tools/grid_pair_submit.py" --root "$GRID_RUN" \
  --release "$GRID_RELEASE" --python "$GRID_PYTHON"
"$GRID_PYTHON" -u - <<'PY_MONITOR'
from pathlib import Path
import json,os,subprocess,time
import sys
sys.path.insert(0,str(Path(os.environ['GRID_RELEASE'])/'tools'))
from grid_pair_monitor import stage_status
r=Path(os.environ['GRID_RUN']); jid=(r/'job_id').read_text().strip()
def lines(path):
 try: rows=path.read_text().splitlines()[-300:]
 except FileNotFoundError: return []
 out=[]
 for row in rows:
  try: out.append(json.loads(row))
  except json.JSONDecodeError: pass
 return out
for attempt in range(2880):
 runs=[]
 queue=json.loads((r/'queue_status.json').read_text()) if (r/'queue_status.json').exists() else {'status':'ENGINEERING_OR_BENCHMARK'}
 sq=subprocess.run(['squeue','-h','-j',jid,'-o','%T|%R'],capture_output=True,text=True,timeout=10)
 scheduler_state=sq.stdout.strip().split('|')[0]
 if not scheduler_state:
  accounting=subprocess.run(['sacct','-X','-n','-P','-j',jid,'-o','State'],capture_output=True,text=True,timeout=10)
  scheduler_state=accounting.stdout.split('|')[0].strip().split()[0] if accounting.stdout.strip() else 'UNKNOWN'
 for log in sorted(r.glob('fold_*/*/training.jsonl')):
  tr=lines(log); dev=lines(log.parent/'development.jsonl'); last=tr[-1] if tr else {}
  candidates=[]
  initial=log.parent/'initial_metrics.json'
  if initial.exists(): candidates.append(dict(json.loads(initial.read_text())['joint'],step=0))
  candidates.extend(dict(row['metrics']['joint'],step=row['step']) for row in dev if not row['startup'])
  engine=json.loads((log.parent/'status.json').read_text()) if (log.parent/'status.json').exists() else {}
  if queue['status']=='PAUSED_DURING_EVALUATION': engine['status']=queue['status']
  state=dict(stage='B2018 inner development',status=stage_status(engine.get('status',''),(log.parent/'results.json').exists(),scheduler_state),
   step=last.get('step',0),updated=last.get('updated'),products={'joint':sorted(candidates,key=lambda x:(x['value'],x['step']))[:1]})
  runs.append(dict(name=log.parent.relative_to(r).as_posix(),status=state,training=tr,development=dev))
 if scheduler_state in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PENDING'):
  queue['status']=scheduler_state
 payload=dict(root=str(r),generated_at=time.time(),runs=runs,full2019=[],read_only=True,
  campaign_queue=dict(campaign_id=r.name,state=queue['status'],jobs=[dict(job_id=jid,state=sq.stdout.strip())]),
  last_update=max((v['status']['updated'] or 0 for v in runs),default=None))
 print('GRID_SNAPSHOT '+json.dumps(payload,allow_nan=False),flush=True)
 if queue['status']=='COMPLETE': break
 if not sq.stdout.strip():
  print('SLURM_FINAL',subprocess.run(['sacct','-j',jid,'-o','JobID,State,ExitCode,Elapsed,AllocTRES'],capture_output=True,text=True).stdout,flush=True)
  for p in sorted((r/'logs').glob('slurm_*.out')): print('LAST_JOB_LOG',p.read_text()[-8000:],flush=True)
  break
 time.sleep(5)
PY_MONITOR
'''
    (folder/'submit.stdin').write_text(script,encoding='utf-8',newline='\n')
    wrapper=('@echo off\r\ntitle GNSS grid69 MLP ResUNet single A800 training\r\n'
        '"%SystemRoot%\\System32\\OpenSSH\\ssh.exe" -T -o ConnectTimeout=20 -o StrictHostKeyChecking=yes '
        '-o ServerAliveInterval=30 -p 10022 u23114@logini.tongji.edu.cn "bash -s" '
        '< "%~dp0submit.stdin" > "%~dp0submit.log"\r\n'
        'echo SSH session ended with code %ERRORLEVEL%. See submit.log.\r\npause\r\n')
    (folder/'submit.cmd').write_text(wrapper,encoding='ascii',newline='')
    write_json(folder/'provenance.json',dict(archive_sha256=record['archive_sha256'],
        release_sha256=record['release_sha256'],gpu='A800',count=1,max_wall_hours=4,
        credentials_stored=False,training_started=False))
    print('AUTHENTICATED_SUBMISSION_FILES_READY',folder,flush=True)


if __name__=='__main__':
    main()
