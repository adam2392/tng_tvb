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
            select_ez_inside

# to run simulation and post processing and data loading
from tvbsim.exp.selectregion import Regions
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

all_patients = ['id001_bt',
    'id002_sd',
    'id003_mg', 'id004_bj', 'id005_ft',
    'id006_mr', 'id007_rd', 'id008_dmc',
    'id009_ba', 'id010_cmn', 'id011_gr',
    'id013_lk', 'id014_vc', 'id015_gjl',
    'id016_lm', 'id017_mk', 'id018_lo', 'id020_lma',
    'id021_jc', 'id022_te', 'id023_br']

# all_patients = ['id001_ac',
#     'id002_cj',
#     'id003_cm', 'id004_cv', 'id005_et',
#     'id006_fb', 'id008_gc',
#     'id009_il', 'id010_js', 'id011_ml', 'id012_pc',
#     'id013_pg', 'id014_rb']

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
    outputdatadir = os.path.join(outputdatadir, patient)
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)

    rawdatadir = os.path.join(metadatadir, patient)

    # define sloader for this patient
    loader = Subject(name=patient, root_dir=rawdatadir, preload=False)
    shuffledpat = None
    # perhaps shuffle connectivity?
    if shuffleweights:
        print("shuffling weights!")
        conn, shuffledpat = process_weights(loader.conn, metadatadir, patient=patient, allpats=all_patients)

    # perform some kind of parameter sweep
    # define the parameter sweeping by changing iext
    # iext_param_sweep = np.arange(2.0,4.0,0.1)
    # iext_param_sweep = [3.0]
    # for i, iext in enumerate(iext_param_sweep):
    # print("Using iext1 value of {}".format(iext))
    for i in range(5):
        # get the ez/pz indices we want to use
        clinezinds = loader.ezinds
        clinpzinds = []
        clinezregions = list(loader.conn.region_labels[clinezinds])
        clinpzregions = []

        ######## SELECT EZ REGIONS OUTSIDE THE CLIN DEFINITIONS
        # if we are sampling regions outside our EZ
        numsamps = 2 # should be around 1-3?
        osr_ezregs, osr_ezinds = select_ez_outside(loader.conn, clinezregions, numsamps)

        ######## SELECT EZ REGIONS INSIDE THE CLIN DEFINITIONS
        # ezregs, ezinds = select_ez_inside(loader.conn, clinezregions, numsamps)

        ######## SET THE MODEL'S EZ AND PZ REGIONS ########
        modelezinds = osr_ezinds
        modelpzinds = []
        modelezregions = osr_ezregs
        modelpzregions = []

        print("Model ez: ", modelezregions, modelezinds)
        print("Model pz: ", modelpzregions, modelpzinds)
        
        ## OUTPUTFILE NAME ##
        filename = os.path.join(outputdatadir,
                    '{0}_dist{1}_{2}.npz'.format(patient, movedist, i))
        metafilename = os.path.join(outputdatadir,
                    '{0}_dist{1}_{2}.json'.format(patient, movedist, i))
        direc, simfilename = os.path.split(filename)
        
        maintvbexp = initialize_tvb_model(loader, ezregions=modelezregions, 
                    pzregions=modelpzregions, period=period) #, Iext=iext)
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
        metadata['shuffledpat'] = shuffledpat

        # save metadata
        loader._writejsonfile(metadata, metafilename)

        # load in the data to run frequency analysis
        # reference = 'monopolar'
        # patdatadir = outputdatadir
        # datafile = filename
        # rawdata, metadata = main_freq.load_raw_data(patdatadir, datafile, metadatadir, patient, reference)

        # mode = 'fft'
        # # create checker for num wins
        # freqoutputdir = os.path.join(freqoutputdatadir, 'freq', mode, patient)
        # if not os.path.exists(freqoutputdir):
        #     os.makedirs(freqoutputdir)
        # # where to save final computation
        # outputfilename = os.path.join(freqoutputdir, 
        #         '{}_{}_{}model.npz'.format(patient, mode, (2*i)))
        # outputmetafilename = os.path.join(freqoutputdir,
        #     '{}_{}_{}meta.json'.format(patient, mode, (2*i)))
        # run_freq_analysis(rawdata, metadata, mode, outputfilename, outputmetafilename)

        # mode = 'morlet'
        # # create checker for num wins
        # freqoutputdir = os.path.join(freqoutputdatadir, 'freq', mode, patient)
        # if not os.path.exists(freqoutputdir):
        #     os.makedirs(freqoutputdir)
        # # where to save final computation
        # outputfilename = os.path.join(freqoutputdir, 
        #         '{}_{}_{}model.npz'.format(patient, mode, (2*i)+1))
        # outputmetafilename = os.path.join(freqoutputdir,
        #     '{}_{}_{}meta.json'.format(patient, mode, (2*i)+1))
        # run_freq_analysis(rawdata, metadata, mode, outputfilename, outputmetafilename)

        '''                 PLOTTING OF DATA                        '''
        # DEFINE FIGURE DIR FOR THIS SIM
        # figdir = os.path.join(outputdatadir, 'fig_'+str(i))
        # # if not os.path.exists(figdir):
        #     # os.makedirs(figdir)

        # config = Config(output_base=figdir) 
        # config.figures.MATPLOTLIB_BACKEND="inline"
        # config.figures.SHOW_FLAG=True
        # plotter = PlotterSim()
        # for idx,key in enumerate(state_vars.keys()):
        #     var = state_vars[key]
        #     if idx==0:
        #         numtime, numsignal = var.shape
        #         ts = np.zeros((len(state_vars.keys()), numtime, numsignal))
        #     ts[idx,...] = var 
        # print(ts.shape)
        # print(TimeseriesDimensions.SPACE.value)

        # # PLOT RAW TS
        # ts_obj = Timeseries(ts, 
        #                 OrderedDict({TimeseriesDimensions.SPACE.value: maintvbexp.conn.region_labels}), 
        #                 times[0], 
        #                 times[1] - times[0], "ms")
        
        # phase_comb = itertools.combinations(state_vars.keys(), 2)
        # for keys in phase_comb:
        #     print("Plotting for ", keys)
        #     print("ONLY PLOTTING THE EZ REGIONS PHASE SPACE")
        #     print(state_vars[keys[0]][modelezinds,:].shape)
        #     keys = list(keys)
        #     data_dict = {
        #         keys[0]: state_vars[keys[0]][modelezinds,:],
        #         keys[1]: state_vars[keys[1]][modelezinds,:],
        #     }
        #     # PLOT THE PHASE PLOTS
        #     special_idx = None

        #     plotter.plot_timeseries(data_dict, [], mode="traj", special_idx=special_idx, 
        #                                 title='Epileptor space trajectory '+' '.join(keys), figure_name="Epileptor Space Trajectory " + ' '.join(keys),
        #                                 labels=maintvbexp.conn.region_labels)

        # print("finished plotting!")
