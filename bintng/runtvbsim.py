import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
sys.path.append('../')
import tvbsim
from tvb.simulator.lab import *
import os.path
import numpy as np
import pandas as pd
# from tvbsim.exp.clinregions import clinregions, outsideclininds

if __name__ == '__main__':
    # read in arguments
    patient = str(sys.argv[1]).lower()
    metadatadir = str(sys.argv[2])
    outputdatadir = str(sys.argv[3])
    movedist = float(sys.argv[4])

    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)

    seegmetadatadir = os.path.join(metadatadir, patient, 'elec')
    tvbmetadatadir = os.path.join(metadatadir, patient, 'tvb')
    tvbsim.util.renamefiles(seegmetadatadir)
    # get the important files
    getmetafile = lambda filename: os.path.join(metadatadir, filename)
    seegfile = os.path.join(seegmetadatadir, 'seeg.txt')
    gainfile = os.path.join(seegmetadatadir, 'gain_inv-square.txt')

    ezhypfile = os.path.join(tvbmetadatadir, 'ez_hypothesis.txt')

    ## OUTPUTFILE NAME ##
    filename = os.path.join(outputdatadir, 
            patient+'_dist' + str(movedist) + '.npz')

    np.random.seed(12345+int(movedist*1000))
    ###################### INITIALIZE TVB SIMULATOR ##################
    # initialize structural connectivity and main simulator object
    connfile = os.path.join(tvbmetadatadir, 'connectivity.zip')
    con = connectivity.Connectivity.from_file(connfile)
    maintvbexp = tvbsim.MainTVBSim(con, condspeed=np.inf)
    # load the necessary data files to run simulation
    maintvbexp.loadseegxyz(seegfile=seegfile)
    maintvbexp.loadgainmat(gainfile=gainfile)
    maintvbexp.loadsurfdata(directory=tvbmetadatadir, use_subcort=False)

    # ezregions, pzregions = clinregions(patient)
    reginds = pd.read_csv(ezhypfile, delimiter='\n').as_matrix()
    ezinds = np.where(reginds==1)[0]
    ezregions = con.region_labels[ezinds]
    pzregions = []

    print(ezregions, pzregions)
    # set ez/pz regions
    maintvbexp.setezregion(ezregions=ezregions)
    maintvbexp.setpzregion(pzregions=pzregions)
    allindices = np.hstack((maintvbexp.ezind, maintvbexp.pzind)).astype(int) 
    # setup models and integrators
    ######### Epileptor Parameters ##########
    epileptor_r = 0.00037#/1.5   # Temporal scaling in the third state variable
    epiks = -10                  # Permittivity coupling, fast to slow time scale
    epitt = 0.5                   # time scale of simulation
    epitau = 10                   # Temporal scaling coefficient in fifth st var
    x0norm=-2.45 # x0c value = -2.05
    x0ez=-1.85
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

    ######################## run simulation ########################
    configs = maintvbexp.setupsim(a=1., period=period, moved=False)
    print(configs)
    times, epilepts, seegts = maintvbexp.mainsim(sim_length=sim_length)

    ######################## POST PROCESSING ########################
    secstoreject = 20

    allindices = np.append(maintvbexp.ezind, maintvbexp.pzind, axis=0).astype(int)
    postprocessor = tvbsim.postprocess.PostProcessor(samplerate=_samplerate, allszindices=allindices)
    times, epits, seegts, zts = postprocessor.postprocts(epilepts, seegts, times, secstoreject=secstoreject)

    # GET ONSET/OFFSET OF SEIZURE
    # postprocessor = tvbsim.postprocess.PostProcessor(samplerate=_samplerate, allszindices=allindices)
    # settimes = postprocessor.getonsetsoffsets(zts, allindices, lookahead=500, delta=0.2)# get the actual seizure times and offsets
    # seizonsets, seizoffsets = postprocessor.getseiztimes(settimes)

    detector = tvbsim.postprocess.detectonsetoffset.DetectShift()
    settimes = detector.getonsetsoffsets(epits, allindices)
    seizonsets, seizoffsets = detector.getseiztimes(settimes)

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
            'ez': maintvbexp.ezregion,
            'pz': maintvbexp.pzregion,
            'ezindices': maintvbexp.ezind,
            'pzindices': maintvbexp.pzind,
            'onsettimes':seizonsets,
            'offsettimes':seizoffsets,
            'patient':patient,
            'samplerate': _samplerate,
            'epiparams': maintvbexp.getepileptorparams(),
            'gainmat': maintvbexp.gainmat
        }
    # save tseries
    np.savez_compressed(filename, epits=epits, seegts=seegts, \
             times=times, zts=zts, metadata=metadata)
