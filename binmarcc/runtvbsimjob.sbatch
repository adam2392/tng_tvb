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

ml anaconda-python/2.7
ml singularity 
source activate tvb

# Debug output to make sure we are in the correct file
echo "Runninng sbatch tvb forward sim file..."
echo ${SBATCH_JOB_NAME}
echo ${SLURM_SUBMIT_DIR}
echo "Patient is: ${patient}"
echo "Metadatadir is: ${metadatadir}"
echo "Outputdatadir is: ${outputdatadir}"
echo "Freqoutputdatadir is: ${freqoutputdatadir}"
echo "Distance to move is: ${dist}"
echo "Should we shuffle weights?: ${shuffleweights}"
echo "Exp name is: ${expname}"

echo "Type of shuffling is: ${typeshuffling}"
echo "Number of simulations to generate is: ${numsims}"
echo "Type of ez selection is: ${ezselectiontype}"

# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER
singularity pull --name tvb.simg shub://adam2392/deeplearning_hubs:tvb

# python ./paramsweep/${expname}/main_sim.py ${patient} --metadatadir ${metadatadir} \
# 	--outputdatadir ${outputdatadir} --freqoutputdatadir ${freqoutputdatadir} \
# 	--movedist ${dist} --shuffleweights ${shuffleweights} 
singularity -d -vvv exec python ./hpc_util/simvsreal/exp_simvsreal.py ${patient} --metadatadir ${metadatadir} \
								--outputdatadir ${outputdatadir} \
								--freqoutputdatadir ${freqoutputdatadir} \
								--movedist ${dist} \
								--shuffleweights ${shuffleweights} \
								--typeshuffling ${typeshuffling} \
								--numsims ${numsims} \
								--ezselectiontype ${ezselectiontype}

exit