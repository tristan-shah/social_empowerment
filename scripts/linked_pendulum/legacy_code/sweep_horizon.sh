#!/bin/bash
#SBATCH --job-name=horizon          # Name of the job
#SBATCH --partition=h100          # Request the h100 partition
#SBATCH --gres=gpu:4              # Request GPU
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --exclusive              # Exclusive node access
#SBATCH --time=24:00:00          # Max runtime
# SBATCH --output=slurm-%j.out    # Output file
# SBATCH --error=slurm-%j.err     # Error file

source ~/miniforge3/etc/profile.d/conda.sh 
conda activate soc_emp
export PYTHONUNBUFFERED=1  # Add this

# Run your Python script
# python scripts/linked_pendulum/sweep_horizon.py --power 1.7 --steps 3000
python scripts/linked_pendulum/sweep_horizon.py --power 200.0