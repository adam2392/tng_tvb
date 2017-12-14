#!/bin/bash

#SBATCH
#SBATCH --mail-type=end
#SBATCH --mail-user=ali39@jhu.edu
#SBATCH --output=_gnuout/%A.out 
#SBATCH --error=_gnuout/%A.err

# Author: Adam Li (ali39@jhu.edu).
# Created on 2017-10-31. 
#---------------------------------------------------------------------
# SLURM job script to run serial Python
# on MARCC using GNU parallel
#---------------------------------------------------------------------

. /soft/miniconda3/activate
source activate tvbforwardsim

# Debug output to make sure we are in the correct file
echo "Runninng sbatch tvb forward sim file..."
echo ${SBATCH_JOB_NAME}
echo ${SLURM_SUBMIT_DIR}
echo ${patient}
echo ${numez}
echo ${numpz}

python runclustersim.py ${patient} ${numez} ${numpz}
