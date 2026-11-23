#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@180
#SBATCH --no-requeue
set -euo pipefail
export GNSS_ALLOCATION_START_EPOCH="$(date +%s)"
: "${GNSS_RELEASE:?}"
: "${GNSS_RETRAIN_ROOT:?}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$GNSS_RELEASE/src:$GNSS_RELEASE/tools:$GNSS_RELEASE/results/20261122_geometry_base_retrain_v01"
srun "$HOME/.conda/envs/ZhouSiyu/bin/python" -u "$GNSS_RELEASE/results/20261122_geometry_base_retrain_v01/train_new_base.py" --mode server
