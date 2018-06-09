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
	'id001_bt')
	# id002_sd
	# id003_mg id004_bj id005_ft
	# id006_mr id007_rd id008_dmc
	# id009_ba id010_cmn id011_gr
	# id013_lk id014_vc id015_gjl
	# id016_lm id017_mk id018_lo')
	# id020_lma')

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

metadatadir="/home/adamli/data/tngpipeline/"
outputdatadir="/home/adamli/data/tvbforwardsim/${expname}/" # and with allregions/
freqoutputdatadir="/home/adamli/data/output/tvbsim/${expname}/"
echo "This is the data directories: "
echo "$metadatadir "
echo "$outputdatadir "

################################### 2. DEFINE SLURM PARAMS ###########################################
# run setup of a slurm job
setup="./config/slurm/setup.sh"
. $setup
# two configuration for slurm type jobs
array_config="./config/slurm/array_jobs.txt"
short_config="./config/slurm/short_jobs.txt"
long_config="./config/slurm/long_jobs.txt"

echo "Running tvb sim ${expname}"

for patient in $patients; do
	echo $patient

	# set jobname
	jobname="${patient}_${expname}_submit_tvbsim.log"
	
	# create export commands
	exvars="--export=patient=${patient},\
metadatadir=${metadatadir},\
outputdatadir=${outputdatadir},\
freqoutputdatadir=${freqoutputdatadir},\
dist=${dist},\
shuffleweights=${shuffleweights} "

	# build basic sbatch command with all params parametrized
	sbatchcomm=$(cat $long_config)
	sbatchcomm="$sbatchcomm --job-name=${jobname}"

	# build a scavenger job, gpu job, or other job
	echo $sbatchcomm $exvars runtvbjob.sbatch 
	echo "Sbatch should run now"
	
	${sbatchcomm} $exvars ./${expname}/runtvbsim_${expname}.sbatch

	read -p "Continuing in 0.5 Seconds...." -t 0.5
	echo "Continuing ...."
done
