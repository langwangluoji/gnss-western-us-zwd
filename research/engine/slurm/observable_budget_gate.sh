#!/bin/bash
#SBATCH --job-name=ZSY_OBS64_gate
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
set -euo pipefail
: "${OBS_RELEASE:?Set OBS_RELEASE}"
: "${OBS_RUN:?Set OBS_RUN}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$OBS_RELEASE/src"
export OBS_ARM=mean
python_bin="${OBS_PYTHON:-$HOME/.conda/envs/ZhouSiyu/bin/python}"
"$python_bin" "$OBS_RELEASE/tools/lock_observable_budget.py" \
  --benchmark "$OBS_RUN/benchmark/benchmark.json" --bundle "$OBS_RELEASE/bundle" \
  --output "$OBS_RUN/budget.lock.json" --per-arm-cap 0.5
# One registered default queue. A failed submission leaves this intent for inspection.
mkdir "$OBS_RUN/.default_queue_submitted"
mean_id=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" \
  --output="$OBS_RUN/logs/mean_%j.out" "$OBS_RELEASE/slurm/observable_mean_a800.sh")
printf '%s\n' "$mean_id" > "$OBS_RUN/mean_job_id"
eval_id=$(sbatch --parsable --dependency="afterok:$mean_id" \
  --output="$OBS_RUN/logs/evaluate_mean_%j.out" "$OBS_RELEASE/slurm/observable_evaluate_a800.sh")
printf '%s\n' "$eval_id" > "$OBS_RUN/evaluation_job_id"
printf 'Registered mean=%s evaluation=%s; maximum one A800 at a time.\n' "$mean_id" "$eval_id"
