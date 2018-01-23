#!/bin/bash
. /soft/miniconda3/activate
source activate tvbforwardsim

# to submit tvb sims
# patients=(
# 	'id001_ac 
# 	id002_cj
# 	id014_rb'
# 	)
patient = 'id001_ac'

# 1. Prompt user for input that runs the analysis
echo "Begin analysis." # print beginning statement
# read -p "Move contacts?: " movecontacts
read -p "Enter num ez: " numez
read -p "Enter num pz: " numpz

# set values and their defauls
# movecontacts=${modelType:-1}
numez=${numez:-1}
numpz=${numpz:-2}

# echo ${movecontacts}
echo ${numez}
echo ${numpz}

# Pause before running to check
printf "About to run on patients (press enter to continue): $patient" 
read answer

metadatadir='/home/adamli/metadata/'
outputdatadir='/home/adamli/data/tvbforwardsim/allregions_moved/' # and with allregions/
printf "\nThis is the data directories: \n"
printf "$metadatadir \n"
printf "$outputdatadir \n"
printf "\n"

################################### 2. DEFINE SLURM PARAMS ###########################################
NUM_PROCSPERNODE=1  	# number of processors per node (1-24). Use 24 for GNU jobs.
NUM_NODES=1				# number of nodes to request
NUM_CPUPERTASK=1

## job reqs
walltime=5:00:0					# the walltime for each computation

# create concatenated strings in unix to ensure proper passing of list of patients
buff=''
for patient in $patients; do
	buff+=$patient
	buff+=' '
done
echo $buff
printf "\n"

#### Create all logging directories if needed
# _logs = the parallel logfile for resuming job at errors 
outdir=_out
logdir='_logs'
# create output directory 
if [ -d "$outdir" ]; then  
	echo "Out log directory exists!\n\n"
else
	mkdir $outdir
fi

printf "Running tvb sim"
for iregion in $(seq 0 83); do
	echo $iregion

	# set jobname
	jobname="${patient}_submit_tvbsim.log"
	
	# create export commands
	exvars="--export=patient=${patient},\
numez=${numez},\
numpz=${numpz},\
metadatadir=${metadatadir},\
outputdatadir=${outputdatadir}, \
iregion=${iregion} "

	# build basic sbatch command with all params parametrized
	sbatcomm="sbatch \
	--time=${walltime} \
	--nodes=${NUM_NODES} \
	--cpus-per-task=${NUM_CPUPERTASK} \
	--job-name=${jobname} "

	# build a scavenger job, gpu job, or other job
	echo $sbatcomm $exvars runtvbjob.sbatch 
	printf "Sbatch should run now\n"
	
	${sbatcomm} $exvars ./runtvbsim.sbatch

	read -p "Continuing in 0.5 Seconds...." -t 0.5
	echo "Continuing ...."
done

# for patient in $patients; do
# 	echo $patient

# 	# set jobname
# 	jobname="${patient}_submit_tvbsim.log"
	
# 	# create export commands
# 	exvars="--export=patient=${patient},\
# numez=${numez},\
# numpz=${numpz},\
# metadatadir=${metadatadir},\
# outputdatadir=${outputdatadir} "

# 	# build basic sbatch command with all params parametrized
# 	sbatcomm="sbatch \
# 	--time=${walltime} \
# 	--nodes=${NUM_NODES} \
# 	--cpus-per-task=${NUM_CPUPERTASK} \
# 	--job-name=${jobname} "

# 	# build a scavenger job, gpu job, or other job
# 	echo $sbatcomm $exvars runtvbjob.sbatch 
# 	printf "Sbatch should run now\n"
	
# 	${sbatcomm} $exvars ./runtvbsim.sbatch
# done
