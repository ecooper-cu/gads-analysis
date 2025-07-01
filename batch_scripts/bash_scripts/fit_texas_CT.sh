#!/bin/bash

#SBATCH --account=ucb678_asc1
#SBATCH --partition=amilan
#SBATCH --job-name=fit_texas_CT
#SBATCH --output=out/ercot_markov_fitting/CT.%j.out
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emco4286@colorado.edu

module purge
module load python

source /projects/emco4286/environments/fitting/bin/activate
python /home/emco4286/gads-analysis/batch_scripts/python_scripts/fit_texas_CT.py