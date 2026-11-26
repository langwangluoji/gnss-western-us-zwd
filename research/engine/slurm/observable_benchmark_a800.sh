#!/bin/bash
#SBATCH --job-name=ZSY_OBS64_bench
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --signal=USR1@120
set -euo pipefail
: "${OBS_RELEASE:?Set OBS_RELEASE to the extracted release directory}"
: "${OBS_RUN:?Set OBS_RUN to the new experiment directory}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$OBS_RELEASE/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="${OBS_PYTHON:-$HOME/.conda/envs/ZhouSiyu/bin/python}"
mkdir -p "$OBS_RUN"
srun "$python_bin" -m gnss_field.training.observable --bundle "$OBS_RELEASE/bundle" \
  --output "$OBS_RUN/benchmark" --arm edm --benchmark
