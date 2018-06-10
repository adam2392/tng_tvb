import os
import sys
import argparse

import numpy as np
import pandas as pd
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
sys.path.append('../')

from tvb.simulator.lab import *
import tvbsim

parser = argparse.ArgumentParser()
parser.add_argument('patient', 
                    help="Patient to analyze")
parser.add_argument('--outputdatadir', default='./',
                    help="Where to save the output simulated data.")
parser.add_argument('--freqoutputdatadir', help="Where to save the output freq analysis of simulated data")
parser.add_argument('--metadatadir', default='/Volumes/ADAM\ LI/rawdata/tngpipeline/',
                    help="Where the metadata for the TVB sims is.")
parser.add_argument('--movedist', default=-1, type=int,
                    help="How to move channels.")
parser.add_argument('--shuffleweights', default=1, type=int,  
                    help="How to move channels.")

all_patients = ['id001_bt',
    'id002_sd',
    'id003_mg', 'id004_bj', 'id005_ft',
    'id006_mr', 'id007_rd', 'id008_dmc',
    'id009_ba', 'id010_cmn', 'id011_gr',
    'id013_lk', 'id014_vc', 'id015_gjl',
    'id016_lm', 'id017_mk', 'id018_lo', 'id020_lma']

def setup_sim():
    pass
    return conn, model, integrator, monitors

def run_sim(conn, model, integrator, monitors):
    pass

    return times, epits, seegts, zts, state_vars

def post_process_data(filename, times, epits, seegts, zts, state_vars):
    print('finished simulating!')
    print(epits.shape)
    print(seegts.shape)
    print(times.shape)
    print(zts.shape)
    print(state_vars.keys())

    # save tseries
    np.savez_compressed(filename, epits=epits, 
                                seegts=seegts,
                                times=times, 
                                zts=zts, 
                                state_vars=state_vars)
    print("saved at {}".format(filename))
    
if __name__ == '__main__':
    args = parser.parse_args()
    # extract passed in variable
    patient = args.patient
    outputdatadir = args.outputdatadir
    metadatadir = args.metadatadir
    movedist = args.movedist
    shuffleweights = args.shuffleweights

    ###### SIMULATION LENGTH AND SAMPLING ######
    # 1000 = 1 second
    samplerate = 1000 # Hz
    sim_length = 180*samplerate    
    period = 1

    conn, model, integrator, monitors = setup_sim()

    times, epits, seegts, zts, state_vars = run_sim(conn, model, integrator, monitors)

    post_process_data(filename, times, epits, seegts, zts, state_vars)
