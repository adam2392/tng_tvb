#!/bin/bash
. /soft/miniconda3/activate
source activate tvbforwardsim

# to submit tvb sims
patients=(
	# 'id001_ac 
	# id002_cj
	# id003_cm
	# id004_cv
	# id005_et
	# id006_fb
	# id008_gc
	# id009_il
	# id010_js
	# id011_ml
	# id012_pc
	# id013_pg
	# id014_rb
	# id015_sf'
# 'id003_cm id008_gc id014_rb'
	'id001_bt
	id002_sd
	id003_mg id004_bj id005_ft
	id006_mr id007_rd id008_dmc
	id009_ba id010_cmn id011_gr
	id013_lk id014_vc id015_gjl
	id016_lm id017_mk id018_lo id020_lma')

# 1. Prompt user for input that runs the analysis
echo "Begin analysis." # print beginning statement
read -p "Enter distance: " dist

# set values and their defauls
dist=${dist:--1}
echo ${dist}

# Pause before running to check
printf "About to run on patients (press enter to continue): $patients" 
read answer

metadatadir='/home/adamli/data/tngpipeline/'
outputdatadir='/home/adamli/data/tvbforwardsim/exp014/' # and with allregions/
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
walltime=12:00:0					# the walltime for each computation

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

printf "Running tvb sim\n"
for patient in $patients; do
	echo $patient

	# set jobname
	jobname="${patient}_submit_tvbsim.log"
	
	# create export commands
	exvars="--export=patient=${patient},\
metadatadir=${metadatadir},\
outputdatadir=${outputdatadir},\
dist=${dist} "

	# build basic sbatch command with all params parametrized
	sbatcomm="sbatch \
	--time=${walltime} \
	--nodes=${NUM_NODES} \
	--cpus-per-task=${NUM_CPUPERTASK} \
	--job-name=${jobname} "

	# build a scavenger job, gpu job, or other job
	echo $sbatcomm $exvars runtvbjob.sbatch 
	printf "Sbatch should run now\n"
	
	${sbatcomm} $exvars ./exp014/runtvbsim_exp014.sbatch

	read -p "Continuing in 0.5 Seconds...." -t 0.5
	echo "Continuing ...."
done