#!/bin/bash
#SBATCH --job-name=ZSY_OBS64_edm_cf
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@120
set -euo pipefail
: "${OBS_RELEASE:?Set OBS_RELEASE}"
: "${OBS_RUN:?Set OBS_RUN}"
: "${OBS_EDM_COVERAGE_REASON:?Register the data-coverage counterfactual before submitting this optional job}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$OBS_RELEASE/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${OBS_PYTHON:-$HOME/.conda/envs/ZhouSiyu/bin/python}"
test -s "$OBS_RUN/budget.lock.json"
srun "$python_bin" -m gnss_field.training.observable --bundle "$OBS_RELEASE/bundle" \
  --output "$OBS_RUN/edm_seed42" --arm edm --seed 42 --budget-lock "$OBS_RUN/budget.lock.json"
