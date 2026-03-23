#!/bin/bash
#SBATCH --job-name=CEM_pendulum
#SBATCH --partition=h200-8-gm1128-c192-m2048
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

source /opt/anaconda/etc/profile.d/conda.sh
conda activate soc_emp
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.8/bin:$PATH
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1

python scripts/linked_pendulum/refactor.py \
    --seed 0 \
    --steps 2000 \
    --alpha 0.01 \
    --horizon 50 \
    --observation_noise 1.0 \
    --stiffness 3.0 \
    --damping 0.1 \
    --state_type angle \
    --control_type egoistic \
    --dt 0.01 \
    --device_batch_size 50 \
    --resolution 100
    