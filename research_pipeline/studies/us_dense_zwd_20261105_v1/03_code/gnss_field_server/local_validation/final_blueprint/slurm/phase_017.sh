#!/bin/bash
#SBATCH --job-name=gnss-017-array
#SBATCH --partition=L40
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@180
#SBATCH --output='E:\Document\FILE_package\Research\SITP\基于生成对抗网络(GAN)的高精度对流层模型构建\research_pipeline\studies\us_dense_zwd_20261105_v1\03_code\gnss_field_server\local_validation\final_blueprint\logs\phase_017_array_%A_%a.out'
#SBATCH --error='E:\Document\FILE_package\Research\SITP\基于生成对抗网络(GAN)的高精度对流层模型构建\research_pipeline\studies\us_dense_zwd_20261105_v1\03_code\gnss_field_server\local_validation\final_blueprint\logs\phase_017_array_%A_%a.err'
#SBATCH --array=0-0%2
set -euo pipefail
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate gnss-field
cd 'E:\Document\FILE_package\Research\SITP\基于生成对抗网络(GAN)的高精度对流层模型构建\research_pipeline\studies\us_dense_zwd_20261105_v1\03_code\gnss_field_server'
srun python -u -m gnss_field worker --task-file 'E:\Document\FILE_package\Research\SITP\基于生成对抗网络(GAN)的高精度对流层模型构建\research_pipeline\studies\us_dense_zwd_20261105_v1\03_code\gnss_field_server\local_validation\final_blueprint\tasks\phase_017.json' --index "${SLURM_ARRAY_TASK_ID}"
