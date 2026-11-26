#!/bin/bash
#SBATCH --job-name=ZSY_OBS64_eval19
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --signal=USR1@120
set -euo pipefail
: "${OBS_RELEASE:?Set OBS_RELEASE}"
: "${OBS_RUN:?Set OBS_RUN}"
arm="${OBS_ARM:-mean}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$OBS_RELEASE/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${OBS_PYTHON:-$HOME/.conda/envs/ZhouSiyu/bin/python}"
test -s "$OBS_RUN/${arm}_seed42/selection.lock.json"
srun "$python_bin" -m gnss_field.training.observable_evaluate --bundle "$OBS_RELEASE/evaluation" \
  --run "$OBS_RUN/${arm}_seed42" --arm "$arm" --output "$OBS_RUN/${arm}_full2019"
