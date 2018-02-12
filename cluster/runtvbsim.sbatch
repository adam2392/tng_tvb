#!/bin/bash

#SBATCH
#SBATCH --mail-type=end
#SBATCH --mail-user=ali39@jhu.edu
#SBATCH --output=_out/%A.out 
#SBATCH --error=_out/%A.err

# Author: Adam Li (ali39@jhu.edu).
# Created on 2017-10-31. 
#---------------------------------------------------------------------
# SLURM job script to run TVB Simulations
#---------------------------------------------------------------------
. /soft/miniconda3/activate
source activate tvbforwardsim

# Debug output to make sure we are in the correct file
echo "Runninng sbatch tvb forward sim file..."
echo ${SBATCH_JOB_NAME}
echo ${SLURM_SUBMIT_DIR}
echo "Patient is: ${patient}"
echo "Metadatadir is: ${metadatadir}"
echo "Outputdatadir is: ${outputdatadir}"
echo "Distance to move is: ${dist}"

python runtvbsim.py ${patient} ${metadatadir} ${outputdatadir} ${dist}
# python runclustersim.py ${patient} ${numez} ${numpz} ${metadatadir} ${outputdatadir} ${dist} ${iregion}
# python runclustersim_overregions.py ${patient} ${numez} ${numpz} ${metadatadir} ${outputdatadir} ${iregion}

exit