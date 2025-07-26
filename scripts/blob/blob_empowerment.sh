#!/bin/bash
#SBATCH --job-name=blob_empowerment          # Name of the job
#SBATCH --partition=h100           # Request the h100 partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --ntasks=1                 # Number of tasks (usually 1 for single-node)
#SBATCH --time=10:00:00            # Max runtime (hh:mm:ss)
#SBATCH --output=output.log        # Standard output and error log

source ~/miniforge3/etc/profile.d/conda.sh 
conda activate soc_emp

# Run your Python script
python scripts/blob_empowerment.py