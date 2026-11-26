"""Create a small, hash-bound overlay on the preserved v02 server release."""
from pathlib import Path
import json
import shutil
import zipfile
from gnss_field.artifacts import read_json, write_json, sha256
from gnss_field.training.release import verify_release

PROJECT = Path(__file__).resolve().parents[1]
NAME = '20261120_observable64_mean_convergence_v01'


def script_header(name, gpu=True, minutes=15):
    return ('#!/bin/bash\n#SBATCH --job-name=ZSY_CV1_'+name+'\n'
        +('#SBATCH --partition=A800\n#SBATCH --gres=gpu:a800:1\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=16G\n'
          if gpu else '#SBATCH --partition=intel\n#SBATCH --cpus-per-task=1\n#SBATCH --mem=2G\n')
        +f'#SBATCH --time=00:{minutes:02d}:00\n'
        +('#SBATCH --signal=USR1@120\n' if gpu else '')
        +'set -euo pipefail\n: "${OBS_RELEASE:?}"\n: "${OBS_RUN:?}"\n'
        +'export PYTHONNOUSERSITE=1\nexport PYTHONPATH="$OBS_RELEASE/src"\nexport CUBLAS_WORKSPACE_CONFIG=:4096:8\n'
        +'python_bin="${OBS_PYTHON:-$HOME/.conda/envs/ZhouSiyu/bin/python}"\n')


def main():
    archive = PROJECT/'dist/20261120_observable64_v02.zip'
    expected = '31016a67516344b5102d01c4e7ad3a33b79620c7a661b75842e5302d730b993b'
    assert sha256(archive) == expected
    stage = PROJECT/'build/obsconv1'
    stage.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        prefix = '20261120_observable_subspace_v02/'
        for member in z.infolist():
            if member.is_dir():
                continue
            assert member.filename.startswith(prefix)
            relative = member.filename[len(prefix):]
            target = stage/relative
            assert target.resolve().is_relative_to(stage.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))
    base = read_json(stage/'release.json')
    assert verify_release(stage) == '0f1d9cdf470910dbec68de61a618dee54d3c55b8c4dae64e590b02af92c91923'
    for relative in ['src/gnss_field/training/observable.py',
                     'src/gnss_field/training/observable_convergence_gate.py', 'tools/lock_observable_budget.py']:
        shutil.copy2(PROJECT/relative, stage/relative)
    old_meta = read_json(stage/'bundle/manifest.json')
    reference = PROJECT/'results/20261120_postcoverage_analysis_v01/artifacts/mean/selected.pt'
    assert sha256(reference) == 'ae04939fc897e1c6c4a2a22e68b5d7581d060d926402eb480cbbf1990b2e7b31'
    (stage/'reference').mkdir()
    shutil.copy2(reference, stage/'reference/mean_selected.pt')
    new_meta = dict(old_meta, maximum_updates_per_arm=5000, convergence_protocol='mean_convergence_v1',
        allowed_arms=['mean'], reference_mean_sha256=sha256(reference),
        reference_bundle_manifest_sha256=sha256(stage/'bundle/manifest.json'))
    write_json(stage/'bundle/manifest.json', new_meta)
    evaluation = read_json(stage/'evaluation/manifest.json')
    evaluation['training_bundle_manifest_sha256'] = sha256(stage/'bundle/manifest.json')
    write_json(stage/'evaluation/manifest.json', evaluation)
    registration = dict(experiment=NAME, maximum_updates=5000, minimum_updates_before_earlystop=1600,
        patience=10, selection='EMA B ES only; raw diagnostic', seed=42, effective_hours=8,
        train_members=4, monitor_members=16, gate_members=64, gate_seed=20261122,
        gate='single selected EMA versus old EMA, paired whole-week 2000 bootstrap upper95 < 0; better than zero',
        baseline_release='0f1d9cdf470910dbec68de61a618dee54d3c55b8c4dae64e590b02af92c91923',
        reference_checkpoint_sha256=sha256(reference),
        plan_sha256=sha256(PROJECT/'docs/superpowers/plans/2026-11-20-mean-convergence-v01.md'),
        stages=['A800 mean engineering benchmark', 'budget lock', 'mean', 'B gate', 'conditional C2019'],
        GPU_wall_caps_minutes=[5, 15, 15], maximum_concurrent_GPU=1,
        automatic_EDM=False, automatic_GAN=False, automatic_retry=False, D2020=False,
        interpretation='New optimization schedule; duration effect not isolated; C is exposed development')
    write_json(stage/'configs/mean_convergence_v01.json', registration)
    scripts = {
        'convergence_benchmark.sh': script_header('bench', minutes=5)+
            'srun "$python_bin" -m gnss_field.training.observable --bundle "$OBS_RELEASE/bundle" --output "$OBS_RUN/benchmark" --arm mean --benchmark\n',
        'convergence_mean.sh': script_header('mean')+
            'srun "$python_bin" -m gnss_field.training.observable --bundle "$OBS_RELEASE/bundle" --output "$OBS_RUN/mean_seed42" --arm mean --seed 42 --budget-lock "$OBS_RUN/budget.lock.json"\n',
        'convergence_evaluate.sh': script_header('B_C_eval')+
            'srun "$python_bin" -m gnss_field.training.observable_convergence_gate --bundle "$OBS_RELEASE/bundle" --run "$OBS_RUN/mean_seed42"\n'
            +'if ! "$python_bin" - "$OBS_RUN/mean_seed42/B_gate.json" <<\'PY_GATE\'\n'
            +'import json,sys\nraise SystemExit(0 if json.load(open(sys.argv[1]))["passed"] else 1)\nPY_GATE\nthen\n'
            +'  printf "SCIENTIFIC_GATE_NOT_PASSED: C2019 skipped; queue stopped.\\n"\n  exit 0\nfi\n'
            +'srun "$python_bin" -m gnss_field.training.observable_evaluate --bundle "$OBS_RELEASE/evaluation" --run "$OBS_RUN/mean_seed42" --arm mean --output "$OBS_RUN/mean_full2019"\n',
        'convergence_budget_gate.sh': script_header('budget', gpu=False, minutes=5)+
            '"$python_bin" "$OBS_RELEASE/tools/lock_observable_budget.py" --benchmark "$OBS_RUN/benchmark/benchmark.json" --bundle "$OBS_RELEASE/bundle" --output "$OBS_RUN/budget.lock.json" --per-arm-cap .25 --evaluation-cap .25\n'
            +'mkdir "$OBS_RUN/.mean_submission_intent"\n'
            +'mean_id=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" --output="$OBS_RUN/logs/mean_%j.out" "$OBS_RELEASE/slurm/convergence_mean.sh")\n'
            +'printf "%s\\n" "$mean_id" > "$OBS_RUN/mean_job_id"\n'
            +'eval_id=$(sbatch --parsable --dependency="afterok:$mean_id" --output="$OBS_RUN/logs/B_C_evaluate_%j.out" "$OBS_RELEASE/slurm/convergence_evaluate.sh")\n'
            +'printf "%s\\n" "$eval_id" > "$OBS_RUN/evaluation_job_id"\n'
            +'printf "MEAN=%s B_C_EVALUATE=%s; no EDM/GAN jobs.\\n" "$mean_id" "$eval_id"\n',
    }
    for filename, content in scripts.items():
        (stage/'slurm'/filename).write_text(content, encoding='utf-8', newline='\n')
    shutil.copy2(PROJECT/'docs/superpowers/plans/2026-11-20-mean-convergence-v01.md', stage/'README.md')
    files = {str(p.relative_to(stage)).replace('\\', '/'): sha256(p) for p in stage.rglob('*')
             if p.is_file() and p.name != 'release.json' and '__pycache__' not in p.parts}
    write_json(stage/'release.json', dict(files=files, study=NAME, no_D2020=True,
        environment='Existing ZhouSiyu; no installs/upgrades', base_release_sha256=registration['baseline_release']))
    release_hash = verify_release(stage)
    changed = [name for name, digest in files.items() if base['files'].get(name) != digest]+['release.json']
    overlay = PROJECT/'dist/20261120_mean_convergence_v01_overlay.zip'
    with zipfile.ZipFile(overlay, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in changed:
            z.write(stage/name, name)
    with zipfile.ZipFile(overlay) as z:
        assert z.testzip() is None
    output = PROJECT/'results/20261120_mean_convergence_v01'
    output.mkdir(exist_ok=True)
    write_json(output/'registration.json', registration)
    write_json(output/'release.json', dict(release_name=NAME, local_stage=str(stage),
        archive=str(overlay), archive_sha256=sha256(overlay), release_sha256=release_hash,
        base_archive_sha256=expected, changed_files=changed, bytes=overlay.stat().st_size,
        submitted=False, D2020=False))
    print('New isolated release', release_hash, 'overlay bytes', overlay.stat().st_size)


if __name__ == '__main__':
    main()
