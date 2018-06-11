import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('./util/')

from tvb.simulator.lab import *
import os.path
import numpy as np
import pandas as pd
import argparse
import itertools

import main_freq

import tvbsim
from tvbsim.postprocess import PostProcessor
from tvbsim.postprocess.detectonsetoffset import DetectShift
from tvbsim.maintvbexp import MainTVBSim
from tvbsim.exp.utils import util
from tvbsim.io.loadsimdataset import LoadSimDataset
from tvbsim.visualize.plotter_sim import PlotterSim
from tvbsim.base.dataobjects.timeseries import TimeseriesDimensions, Timeseries 
from collections import OrderedDict

from tvbsim.base.constants.config import Config


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

def save_processed_data(filename, times, epits, seegts, zts, state_vars):
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
    
def showdebug(maintvbexp):
    sys.stdout.write("The tvbexp ez region is: %s" % maintvbexp.ezregion)
    sys.stdout.write("The tvbexp pz region is: %s" % maintvbexp.pzregion)
    sys.stdout.write("The tvbexp ez indices is: %s" % maintvbexp.ezind)
    sys.stdout.write("The tvbexp pz indices is: %s " % maintvbexp.pzind)


if __name__ == '__main__':
    args = parser.parse_args()

    # extract passed in variable
    patient = args.patient
    outputdatadir = args.outputdatadir
    metadatadir = args.metadatadir
    movedist = args.movedist
    freqoutputdatadir = args.freqoutputdatadir
    shuffleweights = args.shuffleweights

    # define the parameter sweeping by changing iext
    iext_param_sweep = np.arange(2.0,4.0,0.1)
    iext_param_sweep = [3.0]

    simulation parameters
    _factor = 1
    _samplerate = 1000*_factor # Hz
    sim_length = 10*_samplerate    
    period = 1./_factor

    # set all directories to output data, get meta data, get raw data
    outputdatadir = os.path.join(outputdatadir, patient)
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)

    rawdatadir = os.path.join(metadatadir, patient)

    # define loader for this patient
    loader = PatientLoader(patient=patient,
    						rawdatadir=rawdatadir)
    # get the metadata for this patient
    '''
    - connectivity
    - surface
    - ez_hypothesis
    - 
    '''
    INSERT_CODE HERE

    # get the ez/pz indices we want to use
    clinezinds = loader.ezinds
    clinezregions = list(loader.conn.region_labels[clinezinds])
    clinpzregions = []
    allclinregions = clinezregions + clinpzregions

    sys.stdout.write("All clinical regions are: {}".format(allclinregions))
    ###################### INITIALIZE TVB SIMULATOR ##################
    conn = loader.conn
    maintvbexp = MainTVBSim(conn, condspeed=np.inf)
    # load the necessary data files to run simulation
    maintvbexp.loadseegxyz(seegfile=loader.sensorsfile)
    maintvbexp.loadgainmat(gainfile=loader.gainfile)
    maintvbexp.loadsurfdata(directory=loader.tvbdir, use_subcort=False)

    ## OUTPUTFILE NAME ##
    filename = os.path.join(outputdatadir,
                '{0}_dist{1}_{2}.npz'.format(patient, movedist, i))
    metafilename = os.path.join(outputdatadir,
                '{0}_dist{1}_{2}.json'.format(patient, movedist, i))
    direc, simfilename = os.path.split(filename)

    ######### Epileptor Parameters ##########
    epileptor_params = {
        'r': 0.00037,#/1.5   # Temporal scaling in the third state variable
        'Ks': -10,                 # Permittivity coupling, fast to slow time scale
        'tt': 0.07,                   # time scale of simulation
        'tau': 10,                   # Temporal scaling coefficient in fifth st var
        'x0': -2.45, # x0c value = -2.05
        # 'Iext': iext,
    }
    x0ez=-1.65
    x0pz=-2.0 # x0pz = None
    if maintvbexp.ezregion is None:
        x0ez = None
    if maintvbexp.pzregion is None:
        x0pz = None
    maintvbexp.loadepileptor(ezregions=ezregions, pzregions=pzregions,
    					x0ez=x0ez, x0pz=x0pz,**epileptor_params)
    allindices = np.hstack((maintvbexp.ezind, maintvbexp.pzind)).astype(int) 
    show_debug(maintvbexp)

    ######### Integrator Parameters ##########
    n_tau = 0
    noise_cov = np.array([0.001, 0.001, 0.,\
                              0.0001, 0.0001, 0.])
    # define cov noise for the stochastic heun integrator
    hiss = noise.Additive(nsig=noise_cov, ntau=ntau)
    # hiss = noise.Multiplicative(nsig=noise_cov)
    integrator_params = {
    	'dt': 0.05,
    	'noise': hiss,
    }
    maintvbexp.loadintegrator(**integrator_params)

    for ind in maintvbexp.ezind:
            new_seeg_xyz, elecindicesmoved = maintvbexp.move_electrodetoreg(ind, movedist)
        print(elecindicesmoved)
        print(maintvbexp.seeg_labels[elecindicesmoved])

    for i, iext in enumerate(iext_param_sweep):
        print("Using iext1 value of {}".format(iext))
        
        # save metadata from the exp object and from here
        metadata = maintvbexp.get_metadata()
        metadata['patient'] = patient
		metadata['samplerate'] = _samplerate
		metadata['simfilename'] = simfilename
        metadata['clinez'] =clinezregions
        metadata['clinpz'] =clinpzregions

        ######################## run simulation ########################
        initcond = None
        configs = maintvbexp.setupsim(a=1., period=period, 
        						moved=False, initcond=initcond)
        times, statevars_ts, seegts = maintvbexp.mainsim(sim_length=sim_length)

        ######################## POST PROCESSING ########################
        postprocessor = PostProcessor(samplerate=_samplerate, allszindices=allindices)
        secstoreject = 1
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
        loader.savejsondata(metadata, metafilename)

        ''' RUN FREQ DECOMPOSITION '''
        reference = 'monopolar'
        patdatadir = outputdatadir
        datafile = filename
        idx = 0
        winsize = 5000
        stepsize = 2500
        metadata['winsize'] = winsize
        metadata['stepsize'] = stepsize
        
        mode = 'fft'
        # create checker for num wins
        freqoutputdir = os.path.join(freqoutputdatadir, 'freq', mode, patient)
        if not os.path.exists(freqoutputdir):
            os.makedirs(freqoutputdir)
        # where to save final computation
        outputfilename = os.path.join(freqoutputdir, 
                '{}_{}_{}model.npz'.format(patient, mode, idx))
        outputmetafilename = os.path.join(freqoutputdir,
            '{}_{}_{}meta.json'.format(patient, mode, idx))
        rawdata, metadata = main_freq.load_raw_data(patdatadir, datafile, metadatadir, patient, reference)
        main_freq.run_freq(metadata, rawdata, mode, outputfilename, outputmetafilename)

        mode = 'morlet'
        # create checker for num wins
        freqoutputdir = os.path.join(freqoutputdatadir, 'freq', mode, patient)
        if not os.path.exists(freqoutputdir):
            os.makedirs(freqoutputdir)
        # where to save final computation
        outputfilename = os.path.join(freqoutputdir, 
                '{}_{}_{}model.npz'.format(patient, mode, idx))
        outputmetafilename = os.path.join(freqoutputdir,
            '{}_{}_{}meta.json'.format(patient, mode, idx))
        rawdata, metadata = main_freq.load_raw_data(patdatadir, datafile, metadatadir, patient, reference)
        main_freq.run_freq(metadata, rawdata, mode, outputfilename, outputmetafilename)

        idx += 1


        '''                 PLOTTING OF DATA                        '''
        # DEFINE FIGURE DIR FOR THIS SIM
        figdir = os.path.join(outputdatadir, str(i))
        if not os.path.exists(figdir):
            os.makedirs(figdir)

        config = Config(output_base=figdir) 
        config.figures.MATPLOTLIB_BACKEND="inline"
        config.figures.SHOW_FLAG=True
        plotter = PlotterSim()
        for idx,key in enumerate(state_vars.keys()):
            var = state_vars[key]
            if idx==0:
                numtime, numsignal = var.shape
                ts = np.zeros((len(state_vars.keys()), numtime, numsignal))
            ts[idx,...] = var 
        print(ts.shape)
        print(TimeseriesDimensions.SPACE.value)
        # print(maintvbexp.conn.region_labels)

        # PLOT RAW TS
        ts_obj = Timeseries(ts, 
                        OrderedDict({TimeseriesDimensions.SPACE.value: maintvbexp.conn.region_labels}), 
                        times[0], 
                        times[1] - times[0], "ms")
        
        phase_comb = itertools.combinations(state_vars.keys(), 2)
        for keys in phase_comb:
            print("Plotting for ", keys)
            keys = list(keys)
            data_dict = {
                keys[0]: state_vars[keys[0]],
                keys[1]: state_vars[keys[1]]
            }
            # data_dict = {'x1(t)': state_vars['x1'], 
            #         'x2(t)': state_vars['x2'],
            #         'y1(t)': state_vars['y1'],
            #         'y2(t)': state_vars['y2'],
            #         'g(t)': state_vars['g'],
            #         'z(t)': zts}
            # PLOT THE PHASE PLOTS
            special_idx = None

            plotter.plot_timeseries(data_dict, [], mode="traj", special_idx=special_idx, 
                                        title='Epileptor space trajectory '+' '.join(keys), figure_name="Epileptor Space Trajectory " + ' '.join(keys),
                                        labels=maintvbexp.conn.region_labels)

        print("finished plotting!")
