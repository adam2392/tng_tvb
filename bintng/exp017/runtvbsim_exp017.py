import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
sys.path.append('../')
import tvbsim
from tvb.simulator.lab import *
import os.path
import numpy as np
import pandas as pd
import argparse

from tvbsim.exp.utils import util
from tvbsim.io.loadsimdataset import LoadSimDataset

parser = argparse.ArgumentParser()
parser.add_argument('patient', 
                    help="Patient to analyze")
parser.add_argument('--outputdatadir',  
                    help="Where to save the output simulated data.")
parser.add_argument('--metadatadir',  
                    help="Where the metadata for the TVB sims is.")
parser.add_argument('--movedist', default=-1, type=int,
                    help="How to move channels.")
parser.add_argument('--shuffleweights', default=1, type=int,  
                    help="How to move channels.")

all_patients = [
    'id001_bt',
    'id002_sd',
    'id003_mg', 'id004_bj', 'id005_ft',
    'id006_mr', 'id007_rd', 'id008_dmc',
    'id009_ba', 'id010_cmn', 'id011_gr',
    'id013_lk', 'id014_vc', 'id015_gjl',
    'id016_lm', 'id017_mk', 'id018_lo', 'id020_lma']

def post_process_data(filename, times, state_vars_ts, seegts, postprocessor):
    ######################## POST PROCESSING ########################
    secstoreject = 20
    times, epits, seegts, zts, state_vars = postprocessor.postprocts(statevars_ts, seegts, times, secstoreject=secstoreject)

    # loader.load_data(seegts)
    # loader.addlinenoise()
    # loader.filter_data()
    # seegts = loader.rawdata 

    print('finished simulating!')
    print(epits.shape)
    print(seegts.shape)
    print(times.shape)
    print(zts.shape)
    print(state_vars.keys())
    print(allindices)

    # save tseries
    np.savez_compressed(filename, epits=epits, 
                                seegts=seegts,
                                times=times, 
                                zts=zts, 
                                state_vars=state_vars)

if __name__ == '__main__':
    args = parser.parse_args()

    # extract passed in variable
    patient = args.patient
    outputdatadir = args.outputdatadir
    metadatadir = args.metadatadir
    movedist = args.movedist
    shuffleweights = args.shuffleweights

    # set all directories to output data, get meta data, get raw data
    outputdatadir = os.path.join(outputdatadir, patient)
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)

    loader = LoadSimDataset(rawdatadir=metadatadir, patient=patient)

    for i in range(5):
        ###################### INITIALIZE TVB SIMULATOR ##################
        if shuffleweights:
            # within patient shuffling of weights
            conn = connectivity.Connectivity.from_file(loader.connfile)
            randweights = util.randshuffleweights(conn.weights)
            conn.weights = randweights
        else:
            conn = connectivity.Connectivity.from_file(loader.connfile)

        maintvbexp = tvbsim.MainTVBSim(conn, condspeed=np.inf)
        
        # load the necessary data files to run simulation
        maintvbexp.loadseegxyz(seegfile=loader.sensorsfile)
        maintvbexp.loadgainmat(gainfile=loader.gainfile)
        try:
            maintvbexp.loadsurfdata(directory=loader.tvbdir, use_subcort=False)
        except:
            print("Could not load surface data for this patient ", patient)

        # get the ez/pz indices we want to use
        clinezinds = loader.ezinds
        clinezregions = list(conn.region_labels[clinezinds])
        clinpzregions = []
        allclinregions = clinezregions + clinpzregions

        sys.stdout.write("All clinical regions are: {}".format(allclinregions))

        # simulate 3 times with different connectivities
        ## OUTPUTFILE NAME ##
        filename = os.path.join(outputdatadir,
                    '{0}_dist{1}_{2}.npz'.format(patient, movedist, i))
        metafilename = os.path.join(outputdatadir,
                    '{0}_dist{1}_{2}.json'.format(patient, movedist, i))

        # set ez/pz regions
        maintvbexp.setezregion(ezregions=clinezregions)
        maintvbexp.setpzregion(pzregions=clinpzregions)

        sys.stdout.write("The tvbexp ez region is: %s" % maintvbexp.ezregion)
        sys.stdout.write("The tvbexp pz region is: %s" % maintvbexp.pzregion)
        sys.stdout.write("The tvbexp ez indices is: %s" % maintvbexp.ezind)
        sys.stdout.write("The tvbexp pz indices is: %s " % maintvbexp.pzind)
        
        allindices = np.hstack((maintvbexp.ezind, maintvbexp.pzind)).astype(int) 

        # setup models and integrators
        ######### Epileptor Parameters ##########
        epileptor_r = 0.00037#/1.5   # Temporal scaling in the third state variable
        epiks = -10                  # Permittivity coupling, fast to slow time scale
        epitt = 0.05                   # time scale of simulation
        epitau = 10                   # Temporal scaling coefficient in fifth st var
        x0norm=-2.45 # x0c value = -2.05
        x0ez=-1.65
        x0pz=-2.0
        # x0pz = None

        if maintvbexp.ezregion is None:
            x0ez = None
        if maintvbexp.pzregion is None:
            x0pz = None
        ######### Integrator Parameters ##########
        # parameters for heun-stochastic integrator
        heun_ts = 0.05
        noise_cov = np.array([0.001, 0.001, 0.,\
                              0.0001, 0.0001, 0.])
        ntau = 0
        # simulation parameters
        _factor = 1
        _samplerate = 1000*_factor # Hz
        sim_length = 60*_samplerate    
        period = 1./_factor

        maintvbexp.initepileptor(x0norm=x0norm, x0ez=x0ez, x0pz=x0pz,
                                r=epileptor_r, Ks=epiks, tt=epitt, tau=epitau)
        maintvbexp.initintegrator(ts=heun_ts, noise_cov=noise_cov, ntau=ntau)

        for ind in maintvbexp.ezind:
            new_seeg_xyz, elecindicesmoved = maintvbexp.move_electrodetoreg(ind, movedist)
        print(elecindicesmoved)
        print(maintvbexp.seeg_labels[elecindicesmoved])

        # save metadata from the exp object and from here
        metadata = {
                'regions': maintvbexp.conn.region_labels,
                'regions_centers': maintvbexp.conn.centres,
                'chanlabels': maintvbexp.seeg_labels,
                'seeg_xyz': maintvbexp.seeg_xyz,
                'ezregs': maintvbexp.ezregion,
                'pzregs': maintvbexp.pzregion,
                'ezindices': maintvbexp.ezind,
                'pzindices': maintvbexp.pzind,
                'epiparams': maintvbexp.getepileptorparams(),
                'gainmat': maintvbexp.gainmat,
                'x0ez':x0ez,
                'x0pz':x0pz,
                'x0norm':x0norm,
                'patient': patient,
                'samplerate': _samplerate,
                'clinez': clinezregions,
                'clinpz': clinpzregions,
            }

        ######################## run simulation ########################
        initcond = None
        configs = maintvbexp.setupsim(a=1., period=period, moved=False, initcond=initcond)
        times, statevars_ts, seegts = maintvbexp.mainsim(sim_length=sim_length)

        postprocessor = tvbsim.postprocess.PostProcessor(samplerate=_samplerate, allszindices=allindices)
        # save all the raw simulated data
        post_process_data(filename, times, statevars_ts, seegts, postprocessor)

        # GET ONSET/OFFSET OF SEIZURE
        detector = tvbsim.postprocess.detectonsetoffset.DetectShift()
        settimes = detector.getonsetsoffsets(epits, allindices)
        seizonsets, seizoffsets = detector.getseiztimes(settimes)
        print("The detected onset/offsets are: {}".format(zip(seizonsets,seizoffsets)))
        
        metadata['onsettimes'] = seizonsets
        metadata['offsettimes'] = seizoffsets
        # save metadata
        loader.savejsondata(metadata, metafilename)