"""Prepare a single authenticated SSH handoff, without storing any credential."""
import base64
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    project = Path(__file__).resolve().parents[1]
    root = project/'results/20261119_local_edm_subspace_v01'
    record = json.loads((root/'server_release.json').read_text(encoding='utf-8'))
    archive = Path(record['archive'])
    payload = archive.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == record['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    destination = project/'dist/observable_transfer_v02'
    destination.mkdir(exist_ok=True)
    encoded = base64.b64encode(payload).decode('ascii')
    script = '''set -euo pipefail
project_root="$HOME/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117"
python_bin="$HOME/.conda/envs/ZhouSiyu/bin/python"
test -d "$project_root"
test -x "$python_bin"
export OBS_RELEASE="$project_root/code/20261120_observable_subspace_v02"
export OBS_RUN="$project_root/runs/20261120_us_dense_zwd_observable64_coverage_v02"
export OBS_PYTHON="$python_bin"
export OBS_ARM=mean
export PYTHONNOUSERSITE=1
unset OBS_EDM_COVERAGE_REASON
"$python_bin" - <<'PY_UPLOAD'
import base64, hashlib, json, os, zipfile
from pathlib import Path
root=Path.home()/'tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117'
data=base64.b64decode(''' + "'''" + encoded + "'''" + ''')
expected=''' + repr(record['sha256']) + '''
assert hashlib.sha256(data).hexdigest()==expected
archive=root/'uploads/20261120_observable64_v02.zip'
archive.parent.mkdir(exist_ok=True)
if archive.exists():
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected, 'Existing upload differs; preserved'
else:
 temporary=archive.with_suffix('.zip.partial')
 temporary.write_bytes(data)
 os.replace(temporary,archive)
release=Path(os.environ['OBS_RELEASE'])
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 if not release.exists():
  code=root/'code'; code.mkdir(exist_ok=True)
  for member in z.infolist():
   assert (code/member.filename).resolve().is_relative_to(code.resolve())
  z.extractall(code)
manifest=json.loads((release/'release.json').read_text())
for relative, digest in manifest['files'].items():
 assert hashlib.sha256((release/relative).read_bytes()).hexdigest()==digest, relative
print(json.dumps(dict(upload_sha256=expected,release=str(release),files_verified=len(manifest['files']))),flush=True)
PY_UPLOAD
export PYTHONPATH="$OBS_RELEASE/src"
"$python_bin" -c 'import sys,torch,gnss_field; from gnss_field.training.release import verify_release; import os; print(sys.executable,torch.__version__,torch.version.cuda,gnss_field.__file__); print("release",verify_release(os.environ["OBS_RELEASE"]))'
mkdir -p "$OBS_RUN/logs"
squeue -u "$USER" -h -o '%i|%j|%P|%T'
if squeue -u "$USER" -h -t RUNNING,PENDING -o '%P' | grep -qi A800; then
  printf 'An existing A800 job was found; no new jobs submitted. Existing jobs preserved.\n'
  exit 75
fi
for task in observable_benchmark_a800.sh observable_budget_gate.sh observable_mean_a800.sh observable_evaluate_a800.sh; do
  sbatch --test-only "$OBS_RELEASE/slurm/$task"
done
if test -e "$OBS_RUN/benchmark_job_id"; then
  printf 'Existing submission record found; no duplicate submission.\n'
  cat "$OBS_RUN/benchmark_job_id"
  exit 0
fi
mkdir "$OBS_RUN/.bootstrap_submission_intent"
bench_id=$(sbatch --parsable --output="$OBS_RUN/logs/benchmark_%j.out" "$OBS_RELEASE/slurm/observable_benchmark_a800.sh")
printf '%s\n' "$bench_id" > "$OBS_RUN/benchmark_job_id"
gate_id=$(sbatch --parsable --dependency="afterok:$bench_id" --output="$OBS_RUN/logs/budget_gate_%j.out" "$OBS_RELEASE/slurm/observable_budget_gate.sh")
printf '%s\n' "$gate_id" > "$OBS_RUN/gate_job_id"
printf 'Submitted benchmark=%s CPU-budget-gate=%s. No EDM/GAN/search jobs.\n' "$bench_id" "$gate_id"
squeue -j "$bench_id,$gate_id" -o '%.18i %.12P %.24j %.10T %.10M %R'
'''
    (destination/'deploy.stdin').write_text(script, encoding='utf-8', newline='\n')
    wrapper = ('@echo off\r\ntitle GNSS observable v02 A800 handoff\r\n'
        '"%SystemRoot%\\System32\\OpenSSH\\ssh.exe" -T -o ConnectTimeout=15 -o StrictHostKeyChecking=yes '
        '-o ServerAliveInterval=30 -p 10022 u23114@logini.tongji.edu.cn "bash -s" '
        '< "%~dp0deploy.stdin" > "%~dp0deploy.log"\r\n'
        'echo SSH finished with exit code %ERRORLEVEL%.\r\necho See deploy.log in this folder.\r\npause\r\n')
    (destination/'deploy.cmd').write_text(wrapper, encoding='ascii', newline='')
    print('Prepared dist/observable_transfer_v02/deploy.cmd; no credential saved; no connection started.')


if __name__ == '__main__':
    main()
