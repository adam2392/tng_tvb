#!/bin/bash
. /soft/miniconda3/activate
source activate tvbforwardsim

# to submit tvb sims
patients=(
# 	'id001_ac 
# 	id002_cj
# 	id003_cm
# 	id004_cv
# 	id005_et
# 	id006_fb
# 	id008_gc
# 	id009_il
# 	id010_js
# 	id011_ml
# 	id012_pc
# 	id013_pg
# 	id014_rb
# 	id015_sf'
# )
	'id001_bt
	id002_sd
	id003_mg id004_bj id005_ft
	id006_mr id007_rd id008_dmc
	id009_ba id010_cmn id011_gr
	id013_lk id014_vc id015_gjl
	id016_lm id017_mk id018_lo')

# 1. Prompt user for input that runs the analysis
echo "Begin analysis." # print beginning statement
# read -p "Enter num ez: " numez
# read -p "Enter num pz: " numpz
# read -p "Enter ez x0 value: " x0ez
# read -p "Enter pz x0 value: " x0pz

# set values and their defauls
# numez=${numez:-1}
# numpz=${numpz:-1}
# x0ez=${x0ez:--1.6}
# x0pz=${x0pz:--2.0}

# echo ${numez}
# echo ${numpz}
# echo ${x0ez}
# echo ${x0pz}

# Pause before running to check
printf "About to run on patients (press enter to continue): $patients" 
read answer

# metadatadir='/home/adamli/data/metadata/'
metadatadir='/home/adamli/data/tngpipeline/'
outputdatadir='/home/adamli/data/tvbforwardsim/traindata/exp002/' # and with allregions/
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
# create output directory 
if [ -d "$outdir" ]; then  
	echo "Out log directory exists!\n\n"
else
	mkdir $outdir
fi

printf "Running tvb sim\n"
for patient in $patients; do
	echo $patient

	dist=-1
	echo $dist
	# set jobname
	jobname="${patient}_$dist_submit_tvbsim.log"

	for dist in $(seq -1 2 15); do
		echo $dist
		# set jobname
		jobname="${patient}_${dist}_submit_tvbsim.log"
		
		# create export commands
		exvars="--export=patient=${patient},\
metadatadir=${metadatadir},\
outputdatadir=${outputdatadir},\
dist=${dist} "
# x0ez=${x0ez},\
# x0pz=${x0pz},\

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
done


# for x0ez in $(seq -1.6 -0.025 -1.8); do
# 	echo $x0ez

# 	# set jobname
# 	jobname="${patient}_submit_${x0ez}_${x0pz}_tvbsim.log"
	
# 	# create export commands
# 	exvars="--export=patient=${patient},\
# x0ez=${x0ez},\
# x0pz=${x0pz},\
# metadatadir=${metadatadir},\
# outputdatadir=${outputdatadir},\
# dist=${dist} "

# 	# build basic sbatch command with all params parametrized
# 	sbatcomm="sbatch \
# 	--time=${walltime} \
# 	--nodes=${NUM_NODES} \
# 	--cpus-per-task=${NUM_CPUPERTASK} \
# 	--job-name=${jobname} "

# 	# build a scavenger job, gpu job, or other job
# 	echo $sbatcomm $exvars ./runtvbsim.sbatch
# 	printf "Sbatch should run now\n"
	
# 	${sbatcomm} $exvars ./runtvbsim.sbatch

# 	read -p "Continuing in 0.5 Seconds...." -t 0.5
# 	echo "Continuing ...."
# done
