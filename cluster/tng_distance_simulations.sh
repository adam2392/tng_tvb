#!/bin/bash
. /soft/miniconda3/activate
source activate tvbforwardsim

# to submit tvb sims
patients=(
	# 'id001_ac 
	# id002_cj
	# id014_rb
	# 'id008_gc 
	'id013_pg'
	)

# 1. Prompt user for input that runs the analysis
echo "Begin analysis." # print beginning statement
# read -p "Move contacts?: " movecontacts
# read -p "Enter num ez: " numez
# read -p "Enter num pz: " numpz
# read -p "Enter ez x0 value: " x0ez
# read -p "Enter pz x0 value: " x0pz

# set values and their defauls
# movecontacts=${modelType:-1}
# numez=${numez:-1}
# numpz=${numpz:-1}
# x0ez=${x0ez:--1.6}
# x0pz=${x0pz:--2.0}

# echo ${movecontacts}
# echo ${numez}
# echo ${numpz}
# echo ${x0ez}
# echo ${x0pz}

# Pause before running to check
printf "About to run on patients (press enter to continue): $patients" 
read answer

metadatadir='/home/adamli/data/metadata/'
outputdatadir='/home/adamli/data/tvbforwardsim/traindata/full/' # and with allregions/
printf "\nThis is the data directories: \n"
printf "$metadatadir \n"
printf "$outputdatadir \n"
printf "\n"

# create output directory 
if [ -d "$outputdatadir" ]; then  
	echo "output data dir exists!\n\n"
else
	mkdir $outputdatadir
fi

################################### 2. DEFINE SLURM PARAMS ###########################################
NUM_PROCSPERNODE=1  	# number of processors per node (1-24). Use 24 for GNU jobs.
NUM_NODES=1				# number of nodes to request
NUM_CPUPERTASK=1

## job reqs
walltime=2:00:0					# the walltime for each computation

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

# build basic sbatch command with all params parametrized
sbatcomm="sbatch \
--time=${walltime} \
--nodes=${NUM_NODES} \
--cpus-per-task=${NUM_CPUPERTASK} \
--job-name=${jobname} "

printf "Running tvb sim\n"
for patient in $patients; do
	echo $patient

	dist=-1
	echo $dist
	# set jobname
	jobname="${patient}_$dist_submit_tvbsim.log"
		
		# create export commands
	exvars="--export=patient=${patient},\
metadatadir=${metadatadir},\
outputdatadir=${outputdatadir},\
dist=${dist} "
	# build a scavenger job, gpu job, or other job
	echo $sbatcomm $exvars runtvbsim.sbatch 
	printf "Sbatch should run now\n"
	${sbatcomm} $exvars ./runtvbsim.sbatch
	read -p "Continuing in 0.5 Seconds...." -t 0.5
	echo "Continuing ...."

	for dist in $(seq -1 2 15); do
		echo $dist
		# set jobname
		jobname="${patient}_$dist_submit_tvbsim.log"
		
		# create export commands
		exvars="--export=patient=${patient},\
metadatadir=${metadatadir},\
outputdatadir=${outputdatadir},\
dist=${dist} "
# x0ez=${x0ez},\
# x0pz=${x0pz},\

		# build a scavenger job, gpu job, or other job
		echo $sbatcomm $exvars runtvbjob.sbatch 
		printf "Sbatch should run now\n"
		
		${sbatcomm} $exvars ./runtvbsim.sbatch

		read -p "Continuing in 0.5 Seconds...." -t 0.5
		echo "Continuing ...."
	done
done