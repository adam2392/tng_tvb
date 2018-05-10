#!/bin/bash

#SBATCH
#SBATCH --mail-type=end
#SBATCH --mail-user=ali39@jhu.edu
#SBATCH --output=_out/%A.out 
#SBATCH --error=_out/%A.err

# Author: Adam Li (ali39@jhu.edu).
# Created on 2017-10-31. 
#---------------------------------------------------------------------
# SLURM job script to run serial Python
# on MARCC using GNU parallel
#---------------------------------------------------------------------

ml python
ml anaconda-python/2.7

source activate tvb
# . /soft/miniconda3/activate
# source activate tvbforwardsim

# Debug output to make sure we are in the correct file
echo "Runninng sbatch tvb forward sim file..."
echo ${SBATCH_JOB_NAME}
echo ${SLURM_SUBMIT_DIR}
echo ${patient}
echo ${numez}
echo ${numpz}
echo ${metadatadir}
echo ${outputdatadir}
echo ${numprocs}
# echo ${movecontacts}

# for iwin in $(seq 1 $numprocs); do
python runclustersim.py ${patient} ${numez} ${numpz} ${metadatadir} ${outputdatadir} ${numprocs}
# done
exit