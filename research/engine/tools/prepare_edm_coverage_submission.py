"""Register and submit the existing matched EDM arm; production release stays frozen."""
import json
import hashlib
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    previous = project/'results/20261119_local_edm_subspace_v01'
    evidence = json.loads((previous/'server_results_20261120.json').read_text(encoding='utf-8'))
    verified = json.loads((previous/'verification_v02.json').read_text(encoding='utf-8'))
    assert evidence['status'] == 'COMPLETE' and verified['A800_exact_resume_passed']
    assert evidence['benchmark']['real_slurm_signal_resume']
    registration = dict(experiment='20261120_observable64_edm_coverage_cf01',
        purpose='One matched weather-coverage counterfactual: EDM versus trained mean-only Gaussian; test shape gain separately from location shift',
        plan_sha256=hashlib.sha256((project/'docs/superpowers/plans/2026-11-20-edm-coverage-counterfactual.md').read_bytes()).hexdigest(),
        release_sha256=verified['release_sha256'],
        bundle_manifest_sha256=evidence['benchmark']['bundle_manifest_sha256'],
        reference_checkpoint_sha256=evidence['selected']['checkpoint_sha256'],
        reference_RMSE_mm=evidence['metrics']['point']['station_macro_rmse_mm'],
        reference_group_ES_mm=evidence['metrics']['group_macro_ES_mm'],
        training_role='B2018',training_hours=6288,development_anchors=288,seed=42,effective_hours=8,
        maximum_updates=1572,train_members=4,development_members=16,evaluation_members=128,
        sampler_steps=18,EMA=.999,learning_rate=1e-4,
        checkpoint_selection='B development ES only, including zero candidate',
        point_estimator='finite128 members; independently integrated latent mean evaluated separately',
        maximum_additional_training_card_hours=.5,maximum_additional_evaluation_card_hours=.5,
        maximum_concurrent_GPU=1,D2020=False,automatic_expansion=False,
        frozen=['M0','IDW_mix_weight','g(A)','PCA','V','local_covariance','unresolved_error'])
    local = project/'results/20261120_edm_coverage_cf01'
    local.mkdir(exist_ok=True)
    (local/'registration.json').write_text(json.dumps(registration,ensure_ascii=False,indent=2),encoding='utf-8')
    destination = project/'dist/20261120_observable64_edm_cf01'
    destination.mkdir(exist_ok=True)
    encoded = json.dumps(json.dumps(registration,sort_keys=True),ensure_ascii=True)
    script = '''set -euo pipefail
project_root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
export OBS_RELEASE="$project_root/code/20261120_observable_subspace_v02"
export OBS_RUN="$project_root/runs/20261120_us_dense_zwd_observable64_coverage_v02"
export OBS_PYTHON="$HOME/.conda/envs/ZhouSiyu/bin/python"
export PYTHONPATH="$OBS_RELEASE/src"
export PYTHONNOUSERSITE=1
export OBS_ARM=edm
export OBS_EDM_COVERAGE_REASON='Matched 6288-hour coverage and 1572-update counterfactual against completed mean; separate location and distribution-shape contributions'
date -Is
"$OBS_PYTHON" - <<'PY_REGISTER'
import json, os
from pathlib import Path
from gnss_field.artifacts import read_json, sha256, immutable_json
from gnss_field.training.release import verify_release
r=Path(os.environ['OBS_RUN']); release=Path(os.environ['OBS_RELEASE'])
registration=json.loads(''' + encoded + ''')
assert verify_release(release)==registration['release_sha256']
assert sha256(release/'bundle/manifest.json')==registration['bundle_manifest_sha256']
budget=read_json(r/'budget.lock.json'); benchmark=read_json(r/'benchmark/benchmark.json')
assert sha256(r/'benchmark/benchmark.json')==budget['benchmark_sha256']
assert budget['release_sha256']==registration['release_sha256'] and 'edm' in budget['allowed_arms']
assert budget['maximum_updates_per_arm']==1572 and budget['registered_seed']==42
assert budget['maximum_card_hours_per_arm']==.5 and Path(budget['registered_run_root'])==r
assert benchmark['formal_verification'] and benchmark['real_slurm_signal_resume'] and benchmark['real_100_plus100_equals200']
assert benchmark['release_sha256']==registration['release_sha256']
assert read_json(r/'mean_seed42/status.json')['status']=='COMPLETE'
assert read_json(r/'mean_full2019/status.json')['status']=='COMPLETE'
selected=read_json(r/'mean_seed42/selection.lock.json')
assert selected['checkpoint_sha256']==registration['reference_checkpoint_sha256']==sha256(selected['checkpoint'])
metrics=read_json(r/'mean_full2019/metrics.json')
assert metrics['point']['station_macro_rmse_mm']==registration['reference_RMSE_mm']
assert metrics['group_macro_ES_mm']==registration['reference_group_ES_mm']
registration['reference_C2019_metrics_sha256']=sha256(r/'mean_full2019/metrics.json')
immutable_json(r/'edm_coverage_counterfactual_v01.lock.json',registration)
print('REGISTRATION',json.dumps(registration,sort_keys=True),flush=True)
PY_REGISTER
squeue -u "$USER" -h -o '%i|%j|%P|%T|%R'
if test -s "$OBS_RUN/edm_job_id"; then
  printf 'Existing registered EDM job; no duplicate submission.\n'
  cat "$OBS_RUN/edm_job_id"
  exit 0
fi
test ! -e "$OBS_RUN/.observable_gpu.lock"
test ! -e "$OBS_RUN/edm_seed42"
if squeue -u "$USER" -h -t RUNNING,PENDING -o '%P' | grep -qi A800; then
  printf 'Existing A800 job found; existing work preserved and no new job submitted.\n'
  exit 75
fi
sbatch --test-only "$OBS_RELEASE/slurm/observable_edm_a800.sh"
sbatch --test-only "$OBS_RELEASE/slurm/observable_evaluate_a800.sh"
mkdir "$OBS_RUN/.edm_cf01_submission_intent"
edm_id=$(sbatch --parsable --output="$OBS_RUN/logs/edm_cf01_%j.out" "$OBS_RELEASE/slurm/observable_edm_a800.sh")
printf '%s\n' "$edm_id" > "$OBS_RUN/edm_job_id"
eval_id=$(sbatch --parsable --dependency="afterok:$edm_id" --output="$OBS_RUN/logs/evaluate_edm_cf01_%j.out" "$OBS_RELEASE/slurm/observable_evaluate_a800.sh")
printf '%s\n' "$eval_id" > "$OBS_RUN/edm_evaluation_job_id"
printf 'Submitted EDM=%s evaluation=%s; maximum one A800.\n' "$edm_id" "$eval_id"
for attempt in $(seq 1 36); do
  if "$OBS_PYTHON" - "$OBS_RUN" <<'PY_PROGRESS'
from pathlib import Path
import json,sys,math
r=Path(sys.argv[1]); p=r/'edm_seed42/status.json'
if not p.exists():
 print('EDM_NOT_STARTED',flush=True); raise SystemExit(1)
s=json.loads(p.read_text()); print('EDM_STATUS',json.dumps(s),flush=True)
rows=s.get('products',{}).get('joint',[])
if s['step']>=100 and any(row['step']>0 and math.isfinite(row['value']) for row in rows):
 print('FIRST_FINITE_DEVELOPMENT_EVALUATION_CONFIRMED',flush=True)
if s['status'] in ('PAUSED','BUDGET_EXHAUSTED'):
 print('QUEUE_STOPPED_AT_SAVED_BOUNDARY',flush=True); raise SystemExit(0)
ep=r/'edm_full2019/status.json'
if ep.exists():
 e=json.loads(ep.read_text()); print('EDM_EVALUATION_STATUS',json.dumps(e),flush=True)
 if e['status']=='COMPLETE':
  print('EDM_C2019_METRICS',(r/'edm_full2019/metrics.json').read_text(),flush=True)
  print('EDM_SELECTION',(r/'edm_seed42/selection.lock.json').read_text(),flush=True)
  raise SystemExit(0)
raise SystemExit(1)
PY_PROGRESS
  then break; fi
  sleep 10
done
sacct -X -j "$edm_id,$eval_id" -o JobID,JobName%28,Partition,State,ExitCode,Elapsed,Start,End -P
squeue -j "$edm_id,$eval_id" -o '%.18i %.12P %.24j %.10T %.10M %R'
'''
    (destination/'submit.stdin').write_text(script,encoding='utf-8',newline='\n')
    wrapper = ('@echo off\r\ntitle GNSS EDM coverage cf01 A800 submission\r\n'
        '"%SystemRoot%\\System32\\OpenSSH\\ssh.exe" -T -o ConnectTimeout=15 -o StrictHostKeyChecking=yes '
        '-o ServerAliveInterval=30 -p 10022 u23114@logini.tongji.edu.cn "bash -s" '
        '< "%~dp0submit.stdin" > "%~dp0submit.log"\r\n'
        'echo SSH submission finished with exit code %ERRORLEVEL%.\r\necho See submit.log.\r\npause\r\n')
    (destination/'submit.cmd').write_text(wrapper,encoding='ascii',newline='')
    print('Registered matched EDM counterfactual; prepared SSH submission; no production file changed.')


if __name__ == '__main__':
    main()
