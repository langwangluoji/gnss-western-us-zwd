#!/bin/bash
#SBATCH --job-name=gnss-000-array
#SBATCH --partition=A800
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=07:45:00
#SBATCH --signal=USR1@180
#SBATCH --output=/share/home/u23114/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117/runs/20261118_us_dense_zwd_a800_m0_5k_evalfix_v02/logs/phase_000_array_%A_%a.out
#SBATCH --error=/share/home/u23114/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117/runs/20261118_us_dense_zwd_a800_m0_5k_evalfix_v02/logs/phase_000_array_%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:a800:1
set -euo pipefail
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate ZhouSiyu
cd /share/home/u23114/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117/code/a800_evalfix_v04/gnss_field_server
srun python -u -m gnss_field worker --task-file /share/home/u23114/tj23114/packages/dachuang_ztd/ZhouSiyu_DF_GAN_20261117/runs/20261118_us_dense_zwd_a800_m0_5k_evalfix_v02/tasks/phase_000.json --index "${SLURM_ARRAY_TASK_ID}"
