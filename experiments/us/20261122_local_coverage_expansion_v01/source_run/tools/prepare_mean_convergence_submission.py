"""Prepare one authenticated, isolated deployment; never store credentials."""
import base64
import json
from pathlib import Path
from gnss_field.artifacts import read_json, sha256


def main():
    project = Path(__file__).resolve().parents[1]
    record = read_json(project/'results/20261120_mean_convergence_v01/release.json')
    archive = Path(record['archive'])
    assert sha256(archive) == record['archive_sha256']
    payload = base64.b64encode(archive.read_bytes()).decode('ascii')
    folder = project/'dist/20261120_mean_convergence_v01_submission'
    folder.mkdir(exist_ok=True)
    script = '''set -euo pipefail
project_root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
export OBS_PYTHON="$HOME/.conda/envs/ZhouSiyu/bin/python"
export OBS_RELEASE="$project_root/code/20261120_observable64_mean_convergence_v01"
export OBS_RUN="$project_root/runs/20261120_us_dense_zwd_mean_convergence_v01"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$project_root/code/20261120_observable_subspace_v02/src"
date -Is
"$OBS_PYTHON" - <<'PY_DEPLOY'
import base64, hashlib, json, os, shutil, zipfile
from pathlib import Path
from gnss_field.training.release import verify_release
root=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117'
old=root/'code/20261120_observable_subspace_v02'
assert verify_release(old)=='0f1d9cdf470910dbec68de61a618dee54d3c55b8c4dae64e590b02af92c91923'
data=base64.b64decode('''+repr(payload)+''')
assert hashlib.sha256(data).hexdigest()=='''+repr(record['archive_sha256'])+'''
archive=root/'uploads/20261120_mean_convergence_v01_overlay.zip'
if archive.exists():
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==hashlib.sha256(data).hexdigest()
else:
 temporary=archive.with_suffix('.partial'); temporary.write_bytes(data); os.replace(temporary,archive)
release=Path(os.environ['OBS_RELEASE'])
if not release.exists():
 temporary=release.with_name(release.name+'.staging')
 assert not temporary.exists(), 'Existing staging preserved; inspect before retry'
 shutil.copytree(old,temporary,ignore=shutil.ignore_patterns('__pycache__'))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for member in z.infolist():
   assert (temporary/member.filename).resolve().is_relative_to(temporary.resolve())
  z.extractall(temporary)
 assert verify_release(temporary)=='''+repr(record['release_sha256'])+'''
 os.rename(temporary,release)
assert verify_release(release)=='''+repr(record['release_sha256'])+'''
print('ISOLATED_RELEASE_VERIFIED',str(release),flush=True)
PY_DEPLOY
export PYTHONPATH="$OBS_RELEASE/src"
"$OBS_PYTHON" -c 'import sys,torch,gnss_field; print(sys.executable,torch.__version__,gnss_field.__file__)'
mkdir -p "$OBS_RUN/logs"
squeue -u "$USER" -h -o '%i|%j|%P|%T|%R'
if test -e "$OBS_RUN/benchmark_job_id"; then
 printf 'Existing submission preserved; no duplicate submission.\n'; cat "$OBS_RUN/benchmark_job_id"; exit 0
fi
if squeue -u "$USER" -h -t RUNNING,PENDING -o '%P' | grep -qi A800; then
 printf 'Existing A800 work preserved; no new job submitted.\n'; exit 75
fi
for task in convergence_benchmark.sh convergence_budget_gate.sh convergence_mean.sh convergence_evaluate.sh; do
 bash -n "$OBS_RELEASE/slurm/$task"
 sbatch --test-only "$OBS_RELEASE/slurm/$task"
done
mkdir "$OBS_RUN/.bootstrap_submission_intent"
bench_id=$(sbatch --parsable --output="$OBS_RUN/logs/benchmark_%j.out" "$OBS_RELEASE/slurm/convergence_benchmark.sh")
printf '%s\n' "$bench_id" > "$OBS_RUN/benchmark_job_id"
gate_id=$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:$bench_id" --output="$OBS_RUN/logs/budget_%j.out" "$OBS_RELEASE/slurm/convergence_budget_gate.sh")
printf '%s\n' "$gate_id" > "$OBS_RUN/budget_job_id"
printf 'SUBMITTED benchmark=%s budget=%s; only mean, maximum one A800, 35 GPU-min wall caps.\n' "$bench_id" "$gate_id"
for attempt in $(seq 1 48); do
 if "$OBS_PYTHON" - "$OBS_RUN" <<'PY_PROGRESS'
from pathlib import Path
import json,math,sys
r=Path(sys.argv[1]); p=r/'mean_seed42/status.json'
if not p.exists():
 print('WAITING_FOR_ENGINEERING_AND_BUDGET',flush=True); raise SystemExit(1)
s=json.loads(p.read_text()); print('MEAN_STATUS',json.dumps(s),flush=True)
rows=r/'mean_seed42/development.jsonl'
if rows.exists():
 rows=[json.loads(line) for line in rows.read_text().splitlines()]
 if any(row['step']>=100 and math.isfinite(row['metrics']['joint']['value']) for row in rows):
  print('ACTUAL_WEIGHT_UPDATES_AND_FINITE_B_EVALUATION_CONFIRMED',flush=True)
if s['status'] in ('PAUSED','BUDGET_EXHAUSTED'):
 print('SAVED_BOUNDARY_STOP_NO_AUTOMATIC_RETRY',flush=True); raise SystemExit(0)
g=r/'mean_seed42/B_gate.json'
if g.exists():
 gate=json.loads(g.read_text()); print('B_GATE',json.dumps(gate),flush=True)
 if not gate['passed']: raise SystemExit(0)
e=r/'mean_full2019/status.json'
if e.exists():
 status=json.loads(e.read_text()); print('C_EVALUATION_STATUS',json.dumps(status),flush=True)
 if status['status']=='COMPLETE':
  print('C2019_METRICS',(r/'mean_full2019/metrics.json').read_text(),flush=True)
  raise SystemExit(0)
raise SystemExit(1)
PY_PROGRESS
 then break; fi
 sleep 10
done
squeue -u "$USER" -h -o '%i|%j|%P|%T|%R'
cat "$OBS_RUN"/*job_id
'''
    (folder/'submit.stdin').write_text(script, encoding='utf-8', newline='\n')
    wrapper = ('@echo off\r\ntitle GNSS mean convergence v01 A800 submission\r\n'
        '"%SystemRoot%\\System32\\OpenSSH\\ssh.exe" -T -o ConnectTimeout=15 -o StrictHostKeyChecking=yes '
        '-o ServerAliveInterval=30 -p 10022 u23114@logini.tongji.edu.cn "bash -s" '
        '< "%~dp0submit.stdin" > "%~dp0submit.log"\r\n'
        'echo Submission session ended with exit code %ERRORLEVEL%.\r\necho See submit.log.\r\npause\r\n')
    (folder/'submit.cmd').write_text(wrapper, encoding='ascii', newline='')
    print('Prepared isolated mean-only A800 deployment; no connection or job started.')


if __name__ == '__main__':
    main()
