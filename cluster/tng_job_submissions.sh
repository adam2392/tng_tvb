#!/bin/bash

# to submit oarsub -S ./tng_job_submissions

job_name=
NNodes=1
walltime=24:00:00
output=output.%jobid%.out
error=error.%jobid%.out

# run python file for simulating large number of jobs
# python simulate_ez_pz.py