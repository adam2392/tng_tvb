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

# 1. Prompt user for input that runs the analysis
echo "Begin analysis." # print beginning statement
read -p "Enter distance: " dist
read -p "Enter expname: " expname

# set values and their defauls
dist=${dist:--1}
expname=${expname:-exp019-paramiext1sweep}
shuffleweights=${shuffleweights:-1}
echo ${dist}
echo ${expname}

# show 
echo "Parameters read in: $dist $expname"

# Pause before running to check
echo "About to run on patients (press enter to continue): $patients" 
read answer

metadatadir='/home-1/ali39@jhu.edu/data/tngpipeline/'
outputdatadir='/home-1/ali39@jhu.edu/data/tvbforwardsim/'
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
	jobname="${patient}_${numez}_${numpz}_submitgnu_tvbsim.log"

	# create export commands
	exvars="--export=patient=${patient},\
metadatadir=${metadatadir},\
outputdatadir=${outputdatadir},\
freqoutputdatadir=${freqoutputdatadir},\
dist=${dist},\
shuffleweights=${shuffleweights} "

	# build basic sbatch command with all params parametrized
	sbatchcomm=$(cat $long_config)
	sbatchcomm="$sbatchcomm --job-name=${jobname} --partition=${partition}"

	# build a scavenger job, gpu job, or other job
	echo "Sbatch should run now"
	echo $sbatcomm $exvars ./runtvbsimjob.sbatch 

	${sbatcomm} $exvars ./runtvbsimjob.sbatch

	read -p "Continuing in 0.5 Seconds...." -t 0.5
	echo "Continuing ...."
done
# grep for SLURM_EXPORT_ENV when testing