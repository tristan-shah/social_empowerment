#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --partition=h200-8-gm1128-c192-m2048   # partition from your srun earlier
#SBATCH --nodes=1
#SBATCH --gpus=4                                # your server uses --gpus not --gres=gpu:
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G                              # this server has 2048G total, 256G is reasonable for 4 GPUs
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /opt/anaconda/etc/profile.d/conda.sh    # anaconda is at /opt/anaconda on this server
conda activate soc_emp

# Fix for JAX not finding CUDA libs on this server
export LD_LIBRARY_PATH=/users/tashah6/.conda/envs/soc_emp/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$(find /users/tashah6/.conda/envs/soc_emp/lib/python3.10/site-packages/nvidia -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1

python scripts/linked_pendulum/sweep_power.py \
    --seed 0 \
    --steps 2000 \
    --alpha 0.01 \
    --horizon 130 \
    --observation_noise 1.0 \
    --stiffness 3.0 \
    --damping 0.1 \
    --state_type angle \
    --control_type ave \
    --dt 0.01 \
    --device_batch_size 50 \
    --resolution 200 \
    --max_power 4.0