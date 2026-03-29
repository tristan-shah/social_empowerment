#!/bin/bash
#SBATCH --job-name=sweep          # Name of the job
#SBATCH --partition=h100          # Request the h100 partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:4              # Request GPU
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --time=24:00:00          # Max runtime
# SBATCH --output=slurm-%j.out    # Output file
# SBATCH --error=slurm-%j.err     # Error file

source ~/miniforge3/etc/profile.d/conda.sh 
conda activate soc_emp
export PYTHONUNBUFFERED=1  # Add this

# Run your Python script
python scripts/linked_pendulum/sweep_power.py \
    --seed 0 \
    --steps 2000 \
    --alpha 0.01 \
    --horizon 130 \
    --observation_noise 1.0 \
    --stiffness 3.0 \
    --damping 0.1 \
    --state_type angle \
    --control_type egoistic \
    --dt 0.01 \
    --device_batch_size 50 \
    --resolution 100 \
    --max_power 4.0