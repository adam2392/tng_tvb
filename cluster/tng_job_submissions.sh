#!/bin/bash
. /soft/miniconda3/activate
source activate tvbforwardsim

# to submit oarsub -S ./tng_job_submissions
patients=(
	'id001_ac 
	id002_cj
	id014_rb'
	)

# 1. Prompt user for input that runs the analysis
echo "Begin analysis." # print beginning statement
read -p "Move contacts?: " movecontacts
read -p "Enter num ez: " numez
read -p "Enter num pz: " numpz

# set values and their defauls
movecontacts=${modelType:-1}
numez=${numez:-0}
numpz=${numpz:-0}

echo ${movecontacts}
echo ${numez}
echo ${numpz}

# Pause before running to check
printf "About to run on patients (press enter to continue): $patients" 
read answer

metadatadir='/home/adamli/metadata/'
outputdatadir='/home/adamli/data/tvbforwardsim/'
printf "\nThis is the data directories: \n"
printf "$metadatadir \n"
printf "$outputdatadir \n"
printf "\n"

################################### 2. DEFINE SLURM PARAMS ###########################################
NUM_PROCSPERNODE=1  	# number of processors per node (1-24). Use 24 for GNU jobs.
NUM_NODES=1				# number of nodes to request
MEM_NODE=20000 			# GB RAM per node (5-128)
NUM_GPUS=0				# number of GPUS (need 6 procs per gpu)
NUM_CPUPERTASK=1

## job reqs
if [[ "${modelType}" -eq 1 ]]; then
	walltime=5:00:0
else
	walltime=5:00:0					# the walltime for each computationfi
fi

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
logdir='_logs'
# create output directory 
if [ -d "$outdir" ]; then  
	echo "Out log directory exists!\n\n"
else
	mkdir $outdir
fi

printf "Running tvb sim mvar model"
for patient in $patients; do
	echo $patient
# set jobname
	if [[ "${modelType}" -eq 1 ]]; then	
		jobname="${patient}_submit_tvbsim.log"
	else
		jobname="${patient}_submit_tvbsim.log"
	fi
	
	# call python function to compute windows needed for gnu job
	# numWins=`python _preprocessdata.py $patient $winSize $stepSize`
	# printf "\n \n The number of windows to parallelize on is: $numWins \n"
	
	# create export commands
	exvars="--export=patient=${patient},\
numez=${numez},\
numpz=${numpz},\
metadatadir=${metadatadir},\
outputdatadir=${outputdatadir},\
movecontacts=${movecontacts} "

	# build basic sbatch command with all params parametrized
	sbatcomm="sbatch \
	--time=${walltime} \
	--nodes=${NUM_NODES} \
	--cpus-per-task=${NUM_CPUPERTASK} \
	--job-name=${jobname} "
		# --exclusive \
	# --ntasks-per-node=${NUM_PROCSPERNODE} \

	# build a scavenger job, gpu job, or other job
	echo $sbatcomm $exvars runtvbjob.sbatch 
	printf "Sbatch should run now\n"
	
	${sbatcomm} $exvars ./runtvbsim.sbatch
done
