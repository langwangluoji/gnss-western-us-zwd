#!/bin/bash
#SBATCH --job-name=ZSY_gen3_DF_GAN_v02
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --signal=USR1@180
set -euo pipefail
: "${GRID_RELEASE:?}"
: "${GRID_RUN:?}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$GRID_RELEASE/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
srun "${GRID_PYTHON:-$HOME/.conda/envs/ZhouSiyu/bin/python}" -u "$GRID_RELEASE/tools/grid_generation_train.py" \
 --root "$GRID_RUN" --runtime "$GRID_RUN/runtime.json" --mode server-auto --max-seconds "${GRID_ALLOWED_SECONDS:-14040}"
