#!/bin/bash

#SBATCH --partition=amilan
#SBATCH --job-name=fit_texas_type_100_rating_3_dtgrp_1
#SBATCH --output=out/fit_texas_type_100_rating_3_dtgrp_1.%j.out
#SBATCH --time=72:00:00
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emco4286@colorado.edu

source /projects/emco4286/environments/fitting/bin/activate
python /home/emco4286/gads-analysis/batch_scripts/python_scripts/fit_texas_type_100_rating_3_dtgrp_1.py