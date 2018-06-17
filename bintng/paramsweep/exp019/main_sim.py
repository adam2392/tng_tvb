import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
sys.path.append('../')
sys.path.append('./util/')

from tvb.simulator.lab import *
import os.path
import numpy as np
import pandas as pd
import argparse
import itertools

# wrapper for frequency analysis
import main_freq
from util_sim import save_processed_data, process_weights, \
            initialize_tvb_model, showdebug, select_ez_outside, \
            select_ez_inside, run_freq_analysis


# to run simulation and post processing and data loading
from tvbsim.postprocess.postprocess import PostProcessor
from tvbsim.postprocess.detectonsetoffset import DetectShift
from tvbsim.maintvbexp import MainTVBSim
from tvbsim.io.patient.subject import Subject
from tvbsim.base.constants.config import Config

# to run plotting at the end
from tvbsim.visualize.plotter_sim import PlotterSim
from tvbsim.base.dataobjects.timeseries import TimeseriesDimensions, Timeseries 
from collections import OrderedDict

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

# all_patients = ['id001_bt',
#     'id002_sd',
#     'id003_mg', 'id004_bj', 'id005_ft',
#     'id006_mr', 'id007_rd', 'id008_dmc',
#     'id009_ba', 'id010_cmn', 'id011_gr',
#     'id013_lk', 'id014_vc', 'id015_gjl',
#     'id016_lm', 'id017_mk', 'id018_lo', 'id020_lma',
#     'id021', 'id022', 'id023']

all_patients = ['id001_ac',
    'id002_cj',
    'id003_cm', 'id004_cv', 'id005_et',
    'id006_fb', 'id008_gc',
    'id009_il', 'id010_js', 'id011_ml', 'id012_pc',
    'id013_pg', 'id014_rb']


if __name__ == '__main__':
    '''
    MAIN THINGS TO CHANGE:
    1. FOR LOOP FOR PARAMETER SWEEP / MULTIPLE SIMS
    2. PARAMETER INPUTS TO EPILEPTOR MODEL
        - REGIONS,
        - parameter values
    3. shuffling
    '''
    args = parser.parse_args()

    # extract passed in variable
    patient = args.patient
    outputdatadir = args.outputdatadir
    metadatadir = args.metadatadir
    movedist = args.movedist
    freqoutputdatadir = args.freqoutputdatadir
    shuffleweights = args.shuffleweights

    # simulation parameters
    _factor = 1
    _samplerate = 1000*_factor # Hz
    sim_length = 60*_samplerate    
    period = 1./_factor

    # set all directories to output data, get meta data, get raw data
    outputdatadir = os.path.join(outputdatadir, 'clin', patient)
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)

    rawdatadir = os.path.join(metadatadir, patient)

    # define sloader for this patient
    loader = Subject(name=patient, root_dir=rawdatadir, preload=False)
    # perhaps shuffle connectivity?
    # if shuffleweights:
    #     print("shuffling weights!")
    #     conn = process_weights(conn, shuffle=False, patient=None, other_pats=[])

    # perform some kind of parameter sweep
    # define the parameter sweeping by changing iext
    iext_param_sweep = np.arange(2.5,3.5,0.2)
    eps1_param_sweep = np.arange(-0.4,0.4,0.1)
    for i, iext in enumerate(iext_param_sweep):
        for j, eps1 in enumerate(eps1_param_sweep):
            print("Using iext1 value of {}".format(iext))
            print("Using eps1 value of {}".format(eps1))
            
            ######## SELECT EZ REGIONS INSIDE THE CLIN DEFINITIONS
            # get the ez/pz indices we want to use
            clinezinds = loader.ezinds
            clinpzinds = []
            clinezregions = list(loader.conn.region_labels[clinezinds])
            clinpzregions = []

            ezregs, ezinds = select_ez_inside(loader.conn, clinezregions, numsamps=2)

            modelezinds = ezinds
            modelpzinds = []
            modelezregions = ezregs
            modelpzregions = []

            print("Model ez: ", modelezregions, modelezinds)
            print("Model pz: ", modelpzregions, modelpzinds)
                
            ## OUTPUTFILE NAME ##
            filename = os.path.join(outputdatadir,
                        '{0}_dist{1}_{}_{}.npz'.format(patient, movedist, iext, eps1))
            metafilename = os.path.join(outputdatadir,
                        '{0}_dist{1}_{2}_{}.json'.format(patient, movedist, iext, eps1))
            direc, simfilename = os.path.split(filename)
            
            maintvbexp = initialize_tvb_model(loader, ezregions=modelezregions, 
                        pzregions=modelpzregions, period=period, Iext=iext, eps1=eps1)
            allindices = np.hstack((maintvbexp.ezind, maintvbexp.pzind)).astype(int) 
            # move contacts if we wnat to
            for ind in maintvbexp.ezind:
                new_seeg_xyz, elecindicesmoved = maintvbexp.move_electrodetoreg(ind, movedist)
                print(elecindicesmoved)
                print(maintvbexp.seeg_labels[elecindicesmoved])

            # save metadata from the exp object and from here
            metadata = maintvbexp.get_metadata()
            metadata['patient'] = patient
            metadata['samplerate'] = _samplerate
            metadata['simfilename'] = simfilename
            metadata['clinez'] = clinezregions
            metadata['clinpz'] = clinpzregions

            ######################## run simulation ########################
            configs = maintvbexp.setupsim()
            times, statevars_ts, seegts = maintvbexp.mainsim(sim_length=sim_length)

            ######################## POST PROCESSING ########################
            postprocessor = PostProcessor(samplerate=_samplerate, allszindices=allindices)
            secstoreject = 15
            times, epits, seegts, zts, state_vars = postprocessor.postprocts(statevars_ts, seegts, times, secstoreject=secstoreject)
            # save all the raw simulated data
            save_processed_data(filename, times, epits, seegts, zts, state_vars)

            # GET ONSET/OFFSET OF SEIZURE
            detector = DetectShift()
            settimes = detector.getonsetsoffsets(epits, allindices)
            seizonsets, seizoffsets = detector.getseiztimes(settimes)
            print("The detected onset/offsets are: {}".format(zip(seizonsets,seizoffsets)))
            
            metadata['onsettimes'] = seizonsets
            metadata['offsettimes'] = seizoffsets

            # save metadata
            loader._writejsonfile(metadata, metafilename)

            # load in the data to run frequency analysis
            reference = 'monopolar'
            patdatadir = outputdatadir
            datafile = filename
            rawdata, metadata = main_freq.load_raw_data(patdatadir, datafile, metadatadir, patient, reference)

            mode = 'fft'
            # create checker for num wins
            freqoutputdir = os.path.join(freqoutputdatadir, 'freq', mode, patient)
            if not os.path.exists(freqoutputdir):
                os.makedirs(freqoutputdir)
            # where to save final computation
            outputfilename = os.path.join(freqoutputdir, 
                    '{}_{}_{}model.npz'.format(patient, mode, (2*i)))
            outputmetafilename = os.path.join(freqoutputdir,
                '{}_{}_{}meta.json'.format(patient, mode, (2*i)))
            run_freq_analysis(rawdata, metadata, mode, outputfilename, outputmetafilename)

            mode = 'morlet'
            # create checker for num wins
            freqoutputdir = os.path.join(freqoutputdatadir, 'freq', mode, patient)
            if not os.path.exists(freqoutputdir):
                os.makedirs(freqoutputdir)
            # where to save final computation
            outputfilename = os.path.join(freqoutputdir, 
                    '{}_{}_{}model.npz'.format(patient, mode, (2*i)+1))
            outputmetafilename = os.path.join(freqoutputdir,
                '{}_{}_{}meta.json'.format(patient, mode, (2*i)+1))
            run_freq_analysis(rawdata, metadata, mode, outputfilename, outputmetafilename)
