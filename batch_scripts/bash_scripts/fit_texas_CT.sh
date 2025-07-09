#!/bin/bash

#SBATCH --account=ucb678_asc1
#SBATCH --partition=aa100
#SBATCH --job-name=fit_texas_CT
#SBATCH --output=out/ercot_markov_fitting/CT.%j.out
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emco4286@colorado.edu

module purge
module load python/3.10.2
module load cuda/11.8

# Start GPU monitoring in the background
nvidia-smi -l 300 >> gpu_usage-jobid-${SLURM_JOB_ID}.log &
MONITOR_PID=$!

source /projects/emco4286/environments/fitting310/bin/activate
echo "Running .py script"
CUDA_VISIBLE_DEVICES=0 python /home/emco4286/gads-analysis/batch_scripts/python_scripts/fit_texas_CT.py