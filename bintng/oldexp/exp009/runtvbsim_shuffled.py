import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
sys.path.append('../')
import tvbsim
from tvb.simulator.lab import *
import os.path
import numpy as np
import pandas as pd
import random

def clinregions(patient):
    ''' THE REAL CLINICALLY ANNOTATED AREAS '''
    #001
    if 'id001' in patient:
        ezregions = ['ctx-rh-lateralorbitofrontal', 'ctx-rh-temporalpole']
        pzregions = ['ctx-rh-superiorfrontal', 'ctx-rh-rostralmiddlefrontal', 'ctx-lh-lateralorbitofrontal']
    if 'id002' in patient:
        ezregions = ['ctx-lh-lateraloccipital']
        pzregions = ['ctx-lh-inferiorparietal', 'ctx-lh-superiorparietal']
    if 'id003' in patient:
        ezregions = ['ctx-lh-insula']
        pzregions = ['Left-Putamen', 'ctx-lh-postcentral']
    if 'id004' in patient: 
        ''' '''
        ezregions = ['ctx-lh-posteriorcingulate', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-superiorfrontal']
        pzregions = ['ctx-lh-precentral', 'ctx-lh-postcentral']
    if 'id005' in patient: 
        ''' '''
        ezregions = ['ctx-lh-posteriorcingulate', 'ctx-lh-precuneus']
        pzregions = ['ctx-lh-postcentral', 'ctx-lh-superiorparietal']
    if 'id006' in patient: 
        ''' '''
        ezregions = ['ctx-rh-precentral']
        pzregions = ['ctx-rh-postcentral', 'ctx-rh-superiorparietal']
    if 'id007' in patient: 
        ''' '''
        ezregions = ['Right-Amygdala', 'ctx-rh-temporalpole', 'ctx-rh-lateralorbitofrontal']
        pzregions = ['Right-Hippocampus', 'ctx-rh-entorhinal', 'ctx-rh-medialorbitofrontal',
                 'ctx-rh-inferiortemporal', 'ctx-rh-temporalpole', 'ctx-rh-lateralorbitofrontal']    # 008
    if 'id008' in patient:
        ezregions = ['Right-Amygdala', 'Right-Hippocampus']
        pzregions = ['ctx-rh-superiortemporal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-medialorbitofrontal', 'ctx-rh-lateralorbitofrontal']
    if 'id009' in patient:
        ezregions = ['ctx-rh-lingual', 'ctx-rh-parahippocampal']
        pzregions = ['ctx-rh-lateraloccipital', 'ctx-rh-fusiform', 'ctx-rh-inferiorparietal'] # rlocc, rfug, ripc
    if 'id010' in patient:
           
        ezregions = ['ctx-rh-medialorbitofrontal', 'ctx-rh-frontalpole', 'ctx-rh-rostralmiddlefrontal', 'ctx-rh-parsorbitalis'] #  rmofc, rfp, rrmfg, rpor   
        pzregions = ['ctx-rh-lateralorbitofrontal', 'ctx-rh-rostralmiddlefrontal',
                     'ctx-rh-superiorfrontal', 'ctx-rh-caudalmiddlefrontal'] # rlofc, rrmfc, rsfc, rcmfg
    if 'id011' in patient:
        ezregions = ['Right-Hippocampus', 'Right-Amygdala'] # rhi, ramg
        pzregions = ['Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen',
                 'ctx-rh-insula', 'ctx-rh-entorhinal', 'ctx-rh-temporalpole'] # rth, rcd, rpu, rins, rentc, rtmp
    if 'id012' in patient:
        ezregions = ['Right-Hippocampus', 'ctx-rh-fusiform', 'ctx-rh-entorhinal', 'ctx-rh-temporalpole'] # rhi, rfug, rentc, rtmp
        pzregions = ['ctx-lh-fusiform', 'ctx-rh-inferiorparietal', 'ctx-rh-inferiortemporal',
                 'ctx-rh-lateraloccipital', 'ctx-rh-parahippocampal', 'ctx-rh-precuneus', 'ctx-rh-supramarginal'] # lfug, ripc, ritg, rloc, rphig, rpcunc, rsmg
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
    if 'id015' in patient:
        ezregions = ['ctx-rh-lingual', 'ctx-rh-lateraloccipital', 'ctx-rh-cuneus',
                        'ctx-rh-parahippocampal', 'ctx-rh-superiorparietal', 'ctx-rh-fusiform', 'ctx-rh-pericalcarine'] # rlgg, rloc, rcun, rphig, rspc, rfug, rpc
        pzregions = ['ctx-rh-parahippocampal', 'ctx-rh-superiorparietal', 'ctx-rh-fusiform'] # rphig, rspc, rfug
    return ezregions, pzregions

def randshuffleweights(weights):
    weights = np.random.choice(weights.ravel(), size=weights.shape, replace=False)
    return weights
    
if __name__ == '__main__':
    # read in arguments
    patient = str(sys.argv[1]).lower()
    metadatadir = str(sys.argv[2])
    outputdatadir = str(sys.argv[3])
    movedist = float(sys.argv[4])

    outputdatadir = os.path.join(outputdatadir, patient)
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)

    buffmetadatadir = metadatadir
    metadatadir = os.path.join(metadatadir, patient)
    tvbsim.util.renamefiles(metadatadir)
    # get the important files
    getmetafile = lambda filename: os.path.join(metadatadir, filename)
    seegfile = getmetafile('seeg.txt')
    gainfile = getmetafile('gain_inv-square.txt')

    ''' RANDOMLY SAMPLE FROM ANOTHER PATIENT '''
    patients = ['id001_ac', 'id002_cj', 'id003_cm', 'id004_cv',
            'id005_et', 'id006_fb', 'id008_gc', 'id009_il',
           'id010_js', 'id011_ml', 'id012_pc', 'id013_pg', 'id014_rb']
    patsamples = list(patients)
    patsamples.remove(patient)
    randpat = random.choice(patsamples)
    # initialize structural connectivity and main simulator object
    con = connectivity.Connectivity.from_file(os.path.join(buffmetadatadir, randpat, "connectivity.zip"))

    ###################### INITIALIZE TVB SIMULATOR ##################
    # initialize structural connectivity and main simulator object
    con = connectivity.Connectivity.from_file(getmetafile("connectivity.zip"))
    maintvbexp = tvbsim.MainTVBSim(con, condspeed=np.inf)
    # load the necessary data files to run simulation
    maintvbexp.loadseegxyz(seegfile=seegfile)
    maintvbexp.loadgainmat(gainfile=gainfile)
    maintvbexp.loadsurfdata(directory=metadatadir, use_subcort=False)

    ezregions, pzregions = clinregions(patient)
    allclinregions = ezregions + pzregions
    for idx, ezregion in enumerate(allclinregions):
        
        ## OUTPUTFILE NAME ##
        filename = os.path.join(outputdatadir,
                    patient+'_dist' + str(movedist) +   '_' + str(idx) + '.npz')
        print('file to be saved is: ', filename)
        # set ez/pz regions
        maintvbexp.setezregion(ezregions=[ezregion])
        maintvbexp.setpzregion(pzregions=[])

        allindices = np.hstack((maintvbexp.ezind, maintvbexp.pzind)).astype(int) 
        # allindices = allindices.ravel()

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
        sim_length = 80*_samplerate    
        period = 1./_factor

        maintvbexp.initepileptor(x0norm=x0norm, x0ez=x0ez, x0pz=x0pz,
                                r=epileptor_r, Ks=epiks, tt=epitt, tau=epitau)
        maintvbexp.initintegrator(ts=heun_ts, noise_cov=noise_cov, ntau=ntau)

        for ind in maintvbexp.ezind:
            new_seeg_xyz, elecindicesmoved = maintvbexp.move_electrodetoreg(ind, movedist)
        print(elecindicesmoved)
        print(maintvbexp.seeg_labels[elecindicesmoved])

        ######################## run simulation ########################
        initcond = None
        configs = maintvbexp.setupsim(a=1., period=period, moved=False, initcond=initcond)
        print(configs)
        times, epilepts, seegts = maintvbexp.mainsim(sim_length=sim_length)

        ######################## POST PROCESSING ########################
        secstoreject = 20

        postprocessor = tvbsim.postprocess.PostProcessor(samplerate=_samplerate, allszindices=allindices)
        times, epits, seegts, zts = postprocessor.postprocts(epilepts, seegts, times, secstoreject=secstoreject)

        # GET ONSET/OFFSET OF SEIZURE
        postprocessor = tvbsim.postprocess.PostProcessor(samplerate=_samplerate, allszindices=allindices)
        settimes = postprocessor.getonsetsoffsets(zts, allindices, lookahead=100, delta=0.2)# get the actual seizure times and offsets
        seizonsets, seizoffsets = postprocessor.getseiztimes(settimes)

        freqrange = [0.1, 499]
        # linefreq = 60
        noisemodel = tvbsim.postprocess.filters.FilterLinearNoise(samplerate=_samplerate)
        seegts = noisemodel.filter_rawdata(seegts, freqrange)
        # seegts = noisemodel.notchlinenoise(seegts, freq=linefreq)
        print(zip(seizonsets,seizoffsets))

        metadata = {
                'x0ez':x0ez,
                'x0pz':x0pz,
                'x0norm':x0norm,
                'regions': maintvbexp.conn.region_labels,
                'regions_centers': maintvbexp.conn.centres,
                'chanlabels': maintvbexp.seeg_labels,
                'seeg_xyz': maintvbexp.seeg_xyz,
                'ezregs': maintvbexp.ezregion,
                'pzregs': maintvbexp.pzregion,
                'ezindices': maintvbexp.ezind,
                'pzindices': maintvbexp.pzind,
                'onsettimes':seizonsets,
                'offsettimes':seizoffsets,
                'patient':patient,
                'randpat': randpat,
                'samplerate': _samplerate,
                'epiparams': maintvbexp.getepileptorparams(),
                'gainmat': maintvbexp.gainmat
            }
        # save tseries
        np.savez_compressed(filename, epits=epits, seegts=seegts, \
                 times=times, zts=zts, metadata=metadata)
