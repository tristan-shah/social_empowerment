#!/bin/bash
#SBATCH --job-name=vicsek
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate soc_emp
export PYTHONUNBUFFERED=1

python scripts/flock/vicsek/main.py \
    --seed 0 \
    --steps 2000 \
    --num_agents 100 \
    --grid_size 5.0 \
    --radius 0.5 \
    --speed 1.0 \
    --J 0.1 \
    --D 0.0 \
    --horizon 5 \
    --power_density 2.0 \
    --alpha 0.01 \
    --observation_noise 1.0 \
    --behavior passive