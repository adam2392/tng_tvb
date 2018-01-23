#!/bin/bash -l

##############################################################
#
# Shell script for submitting parallel python jobs on SLURM 
# cluster with nodes, CPUS, tasks, GPUs
#
##############################################################
module unload git

ml python
ml parallel
ml anaconda-python/2.7

source activate tvb
# patients listed 5 per row
patients=(
    'id001_ac
    id002_cj
    id014_rb'
 )

## activate virtualenv/condaenv to use their modules

# 1. Prompt user for input that runs the analysis
echo "Begin analysis." # print beginning statement
read -p "Enter num ez: " numez
read -p "Enter num pz: " numpz

# set values and their defauls
numez=${numez:-2}
numpz=${numpz:-0}

# NEED TO RUN FOR EZ=0,1,2,3 and varying PZ all once

# show 
echo $numez
echo $numpz

# Pause before running to check
printf "About to run on patients (press enter to continue): $patients" 
read answer

metadatadir='/home-1/ali39@jhu.edu/data/fragility/tvbforwardsim/metadata/'
outputdatadir='/home-1/ali39@jhu.edu/data/fragility/tvbforwardsim/'
printf "\nThis is the data directories: \n"
printf "Metadatadir: $metadatadir \n"
printf "Output datadir: $outputdatadir \n"
printf "\n"


# create concatenated strings in unix to ensure proper passing of list of patients
buff=''
for patient in $patients; do
	buff+=$patient
	buff+=' '
done
echo $buff
printf "\n"

#### Create all logging directories if needed
# _gnuerr = all error logs for sbatch gnu runs %A.out 
# _gnuout = all output logs for sbatch gnu runs %A.out 
# _logs = the parallel gnu logfile for resuming job at errors 
outdir=_out
# create output directory 
if [ -d "$outdir" ]; then  
	echo "Out log directory exists!\n\n"
else
	mkdir $outdir
fi

# 2. Define Slurm Parameters
NUM_PROCSPERNODE=1  	# number of processors per node (1-24). Use 24 for GNU jobs.
NUM_NODES=1				# number of nodes to request
NUM_CPUPERTASK=1

partition=shared 	# debug, shared, unlimited, parallel, gpu, lrgmem, scavenger
# partition=debug
qos=scavenger

# 3. Submit parallel jobs for python
for patient in $patients; do
	echo $patient

	# set jobname
	jobname="${patient}_${numez}_${numpz}_submitgnu_tvbsim.log"


	# create export commands
	exvars="patient=${patient},\
numez=${numez},\
numpz=${numpz},\
metadatadir=${metadatadir},\
outputdatadir=${outputdatadir},\
numprocs=${NUM_PROCSPERNODE} "

	# build basic sbatch command with all params parametrized
	## job reqs
	walltime=2:00:0					# the walltime for each computationfi

	# build basic sbatch command with all params parametrized
	sbatcomm="sbatch \
	 --time=${walltime} \
	 --nodes=${NUM_NODES} \
	 --cpus-per-task=${NUM_CPUPERTASK} \
	 --job-name=${jobname} \
	 --ntasks-per-node=${NUM_PROCSPERNODE} \
	 --partition=${partition} "

	# build a scavenger job, gpu job, or other job
	printf "Sbatch should run now\n"
	
	echo $sbatcomm $exvars ./runtvbsimjob.sbatch 

	${sbatcomm} --export=$exvars ./runtvbsimjob.sbatch

	read -p "Continuing in 0.5 Seconds...." -t 0.5
	echo "Continuing ...."
done
# grep for SLURM_EXPORT_ENV when testing