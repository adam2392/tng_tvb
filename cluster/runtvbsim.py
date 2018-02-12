import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
sys.path.append('../')
import tvbsim
from tvb.simulator.lab import *
import os.path
import numpy as np
import pandas as pd

def clinregions(patient):
    ''' THE REAL CLINICALLY ANNOTATED AREAS '''
    #001
    if 'id001' in patient:
        ezregions = ['ctx-rh-lateralorbitofrontal', 'ctx-rh-temporalpole']
        pzregions = ['ctx-rh-superiorfrontal', 'ctx-rh-rostralmiddlefrontal', 'ctx-lh-lateralorbitofrontal']

    # 008
    if 'id008' in patient:
        ezregions = ['Right-Amygdala', 'Right-Hippocampus']
        pzregions = ['ctx-rh-superiortemporal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal',
         'ctx-rh-medialorbitofrontal', 'ctx-rh-lateralorbitofrontal']

    # 013
    if 'id013' in patient:
        ezregions = ['ctx-rh-fusiform']
        pzregions = ['ctx-rh-inferiortemporal','Right-Hippocampus','Right-Amygdala', 
              'ctx-rh-middletemporal','ctx-rh-entorhinal']

    # 014
    if 'id014' in patient:
        ezregions = ['Left-Amygdala', 'Left-Hippocampus', 'ctx-lh-entorhinal', 'ctx-lh-fusiform',
             'ctx-lh-temporalpole','ctx-rh-entorhinal']
        pzregions = ['ctx-lh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-lh-inferiortemporal',
             'ctx-lh-insula', 'ctx-lh-parahippocampal']
    return ezregions, pzregions

if __name__ == '__main__':
    # read in arguments
    patient = str(sys.argv[1]).lower()
    metadatadir = str(sys.argv[2])
    outputdatadir = str(sys.argv[3])
    movedist = float(sys.argv[4])

    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)

    tvbsim.util.renamefiles(patient, metadatadir)
    metadatadir = os.path.join(metadatadir, patient)
    # get the important files
    getmetafile = lambda filename: os.path.join(metadatadir, filename)
    seegfile = getmetafile('seeg.txt')
    gainfile = getmetafile('gain_inv-square.txt')

    ## OUTPUTFILE NAME ##
    filename = os.path.join(outputdatadir, 
            patient+'_dist' + str(movedist) + '.npz')

    ###################### INITIALIZE TVB SIMULATOR ##################
    # initialize structural connectivity and main simulator object
    con = connectivity.Connectivity.from_file(getmetafile("connectivity.zip"))
    maintvbexp = tvbsim.MainTVBSim(con)
    # load the necessary data files to run simulation
    maintvbexp.loadseegxyz(seegfile=seegfile)
    maintvbexp.loadgainmat(gainfile=gainfile)
    maintvbexp.loadsurfdata(directory=metadatadir, use_subcort=False)

    ezregions, pzregions = clinregions(patient)
    # set ez/pz regions
    maintvbexp.setezregion(ezregions=ezregions)
    maintvbexp.setpzregion(pzregions=[])
    allindices = np.append(maintvbexp.ezind, maintvbexp.pzind, axis=0).astype(int)
    # setup models and integrators
    ######### Epileptor Parameters ##########
    epileptor_r = 0.00035#/1.5   # Temporal scaling in the third state variable
    epiks = -0.5                  # Permittivity coupling, fast to slow time scale
    epitt = 0.05                   # time scale of simulation
    epitau = 10                   # Temporal scaling coefficient in fifth st var
    x0norm=-2.35 # x0c value = -2.05
    x0ez=-1.85
    # x0pz=-2.2
    x0pz = None
    ######### Integrator Parameters ##########
    # parameters for heun-stochastic integrator
    heun_ts = 0.05
    noise_cov = np.array([0.001, 0.001, 0.,\
                          0.0001, 0.0001, 0.])
    # simulation parameters
    _factor = 1
    _samplerate = 1000*_factor # Hz
    sim_length = 80*_samplerate    
    period = 1./_factor

    maintvbexp.initepileptor(x0norm=x0norm, x0ez=x0ez, x0pz=x0pz,
                            r=epileptor_r, Ks=epiks, tt=epitt, tau=epitau)
    maintvbexp.initintegrator(ts=heun_ts, noise_cov=noise_cov)

    for ind in maintvbexp.ezind:
        print(maintvbexp.move_electrodetoreg(ind, movedist))   

    ######################## run simulation ########################
    configs = maintvbexp.setupsim(a=1., period=period, moved=False)
    print(configs)
    times, epilepts, seegts = maintvbexp.mainsim(sim_length=sim_length)

    ######################## POST PROCESSING ########################
    secstoreject = 15

    postprocessor = tvbsim.postprocess.PostProcessor(samplerate=_samplerate, allszindices=allindices)
    times, epits, seegts, zts = postprocessor.postprocts(epilepts, seegts, times, secstoreject=secstoreject)

    # get the onsettimes and offsettimes for ez/pz indices
    postprocessor = tvbsim.postprocess.PostProcessor(samplerate=_samplerate, allszindices=allindices)
    settimes = postprocessor.getonsetsoffsets(zts, allindices, delta=0.2/5)# get the actual seizure times and offsets
    seizonsets, seizoffsets = postprocessor.getseiztimes(settimes)

    freqrange = [0.1, 499]
    # linefreq = 60
    noisemodel = tvbsim.postprocess.filters.FilterLinearNoise(samplerate=_samplerate)
    seegts = noisemodel.filter_rawdata(seegts, freqrange)
    # seegts = noisemodel.notchlinenoise(seegts, freq=linefreq)
    print(seegts.shape)
    print(zip(seizonsets,seizoffsets))

    metadata = {
            'x0ez':x0ez,
            'x0pz':x0pz,
            'x0norm':x0norm,
            'regions': maintvbexp.conn.region_labels,
            'regions_centers': maintvbexp.conn.centres,
            'seeg_contacts': maintvbexp.seeg_labels,
            'seeg_xyz': maintvbexp.seeg_xyz,
            'ez': maintvbexp.ezregion,
            'pz': maintvbexp.pzregion,
            'ezindices': maintvbexp.ezind,
            'pzindices': maintvbexp.pzind,
            'onsettimes':seizonsets,
            'offsettimes':seizoffsets,
            'patient':patient,
            'samplerate': samplerate,
            'epiparams': maintvbexp.getepileptorparams()
        }
    # save tseries
    np.savez_compressed(filename, epits=epits, seegts=seegts, \
             times=times, zts=zts, metadata=metadata)
