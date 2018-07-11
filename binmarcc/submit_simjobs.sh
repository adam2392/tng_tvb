#!/bin/bash -l

##############################################################
#
# Shell script for submitting parallel python jobs on SLURM 
# cluster with nodes, CPUS, tasks, GPUs
#
##############################################################
module unload git
ml anaconda-python/2.7
source activate tvb

# patients listed 5 per row
patients=(
	'id001_ac')
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
	# id014_rb')
	# 'id001_bt
	# id002_sd
	# id003_mg id004_bj id005_ft
	# id006_mr id007_rd id008_dmc
	# id009_ba id010_cmn id011_gr
	# id013_lk id014_vc id015_gjl
	# id016_lm id017_mk id018_lo
	# id020_lma id021_jc id022_te id023_br')
	# 'id003_cm id008_gc id014_rb'

# 1. Prompt user for input that runs the analysis
echo "Begin analysis." # print beginning statement
read -p "Enter distance: " dist
read -p "Enter expname: " expname
read -p "Should we shuffle weights? : " shuffleweights
read -p "Type of shuffling to be done: " typeshuffling
read -p "Number of simulations to run: " numsims
read -p "EZ selection type: " ezselectiontype

# set values and their defauls
dist=${dist:--1}
expname=${expname:-exp026_clin}
shuffleweights=${shuffleweights:-0}
typeshuffling=${typeshuffling:-within}
numsims=${numsims:-1}
ezselectiontype=${ezselectiontype:-clin}

# show 
echo "Parameters read in: $dist $expname $shuffleweights $typeshuffling $numsims $ezselectiontype"

# Pause before running to check
echo "About to run on patients (press enter to continue): $patients" 
read answer

metadatadir="/home-1/ali39@jhu.edu/data/tngpipeline/"
outputdatadir="/home-1/ali39@jhu.edu/data/tvbforwardsim/${expname}/"
freqoutputdatadir="/home-1/ali39@jhu.edu/data/output/tvbsim/${expname}/"
printf "\nThis is the data directories: \n"
printf "Metadatadir: $metadatadir \n"
printf "Output datadir: $outputdatadir \n"
printf "\n"

# run setup of a slurm job
setup="./config/slurm/setup.sh"
. $setup
# two configuration for slurm type jobs
array_config="./config/slurm/array_jobs.txt"
short_config="./config/slurm/short_jobs.txt"
long_config="./config/slurm/long_jobs.txt"

# 2. Define Slurm Parameters
partition=shared 	# debug, shared, unlimited, parallel, gpu, lrgmem, scavenger
# partition=debug
qos=scavenger

# 3. Submit parallel jobs for python
for patient in $patients; do
	echo $patient

	# set jobname
	jobname="${patient}_${expname}_submitgnu_tvbsim.log"

	# create export commands
	exvars="--export=patient=${patient},\
metadatadir=${metadatadir},\
outputdatadir=${outputdatadir},\
freqoutputdatadir=${freqoutputdatadir},\
dist=${dist},\
numsims=${numsims},\
shuffleweights=${shuffleweights},\
typeshuffling=${typeshuffling},\
ezselectiontype=${ezselectiontype} "

	# build basic sbatch command with all params parametrized
	sbatchcomm=$(cat $short_config)
	sbatchcomm="$sbatchcomm --job-name=${jobname} --partition=${partition}"

	# build a scavenger job, gpu job, or other job
	echo "Sbatch should run now"
	echo $sbatchcomm $exvars ./runtvbsimjob.sbatch 
	${sbatchcomm} $exvars ./runtvbsimjob.sbatch

	read -p "Continuing in 0.5 Seconds...." -t 0.5
	echo "Continuing ...."
done
# grep for SLURM_EXPORT_ENV when testing