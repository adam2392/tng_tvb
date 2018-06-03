#!/bin/bash -l

#### Create all logging directories if needed
# _logs = the parallel logfile for resuming job at errors 
outdir=_out
# logdir='_logs'
# create output directory 
if [ -d "$outdir" ]; then  
	echo "Out log directory exists!\n\n"
else
	mkdir $outdir
fi

# create output directory 
# if [ -d "$logdir" ]; then  
# 	echo " data dir exists!\n\n"
# else
# 	mkdir $outputdatadir
# fi

