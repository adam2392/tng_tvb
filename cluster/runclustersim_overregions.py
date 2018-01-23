import sys
sys.path.append('../tvb-library/')
sys.path.append('../tvb-data/')
sys.path.append('../')

from tvb.simulator.lab import *
import os.path
import time
import scipy.signal as sig

import numpy as np
import pandas as pd
import scipy.io

# downloaded library for peak detection in z time series
import peakdetect
from runmainsim import *
import tvbsim

from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y
def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass', analog=False)
    return b, a
def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def getseiztimes(onsettimes, offsettimes):
    minsize = np.min((len(onsettimes),len(offsettimes)))
    seizonsets = []
    seizoffsets = []
    
    # perform some checks
    if minsize == 0:
        print("no full onset/offset available!")
        return 0
    
    idx = 0
    # to store the ones we are checking rn
    _onset = onsettimes[idx]
    _offset = offsettimes[idx]
    seizonsets.append(_onset)
    
    # start loop after the first onset/offset pair
    for i in range(1,minsize):        
        # to store the previoius values
        _nextonset = onsettimes[i]
        _nextoffset = offsettimes[i]
        
        # check this range and add the offset if it was a full seizure
        # before the next seizure
        if _nextonset < _offset:
            _offset = _nextoffset
        else:
            seizoffsets.append(_offset)
            idx = i
            # to store the ones we are checking rn
            _onset = onsettimes[idx]
            _offset = offsettimes[idx]
            seizonsets.append(_onset)
    if len(seizonsets) != len(seizoffsets):
        seizonsets = seizonsets[0:len(seizoffsets)]
    return seizonsets, seizoffsets
            
def getonsetsoffsets(zts, ezindices, pzindices):
    # create lambda function for checking the indices
    check = lambda indices: isinstance(indices,np.ndarray) and len(indices)>=1

    onsettimes=np.array([])
    offsettimes=np.array([])
    if check(ezindices):
        for ezindex in ezindices:
            _onsettimes, _offsettimes = postprocessor.findonsetoffset(zts[ezindex, :].squeeze(), 
                                                                    delta=0.2/8)
            onsettimes = np.append(onsettimes, np.asarray(_onsettimes))
            offsettimes = np.append(offsettimes, np.asarray(_offsettimes))

    if check(pzindices):
        for pzindex in pzindices:
            _onsettimes, _offsettimes = postprocessor.findonsetoffset(zts[pzindex, :].squeeze(), 
                                                                    delta=0.2/8)
            onsettimes = np.append(onsettimes, np.asarray(_onsettimes))
            offsettimes = np.append(offsettimes, np.asarray(_offsettimes))

    # first sort onsettimes and offsettimes
    onsettimes.sort()
    offsettimes.sort()
    
    return onsettimes, offsettimes

def runclustersim(patient,eznum=1,pznum=0,metadatadir=None,outputdatadir=None,iregion=0):
    sys.stdout.write(patient)
    ###### SIMULATION LENGTH AND SAMPLING ######
    # 1000 = 1 second
    samplerate = 1000 # Hz
    sim_length = 120*samplerate    
    period = 1
    movedistance = 0

    ######### Epileptor Parameters ##########
    # intialized hard coded parameters
    epileptor_r = 0.0002#/1.5   # Temporal scaling in the third state variable
    epiks = -1                  # Permittivity coupling, fast to slow time scale
    epitt = 0.025               # time scale of simulation
    epitau = 4                # Temporal scaling coefficient in fifth st var
    
    # x0c value = -2.05
    x0norm=-2.4
    x0ez=-1.7
    x0pz=-2.0

    eznum = 1
    pznum = 3
    # depends on epileptor variables of interest: it is where the x2-y2 var is
    varindex = [1]

    ######### Integrator Parameters ##########
    # parameters for heun-stochastic integrator
    heun_ts = 0.05
    noise_cov = np.array([0.001, 0.001, 0.,\
                    0.0001, 0.0001, 0.])/3

    project_dir = os.path.join(metadatadir, patient)
    outputdir = os.path.join(outputdatadir, patient)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    tvbsim.util.renamefiles(patient, project_dir)

    ####### Initialize files needed to 
    # convert seeg.xyz to seeg.txt file
    sensorsfile = os.path.join(project_dir, "seeg.txt")
    gainmatfile = os.path.join(project_dir, "gain_inv-square.txt")
    confile = os.path.join(project_dir, "connectivity.zip")

    use_subcort = 1
    verts, normals, areas, regmap = tvbsim.util.read_surf(project_dir, use_subcort)

    ####################### 1. Structural Connectivity ########################
    con = tvbsim.initializers.connectivity.initconn(confile)
    
    # extract the seeg_xyz coords and the region centers
    seeg_xyz = tvbsim.util.extractseegxyz(sensorsfile)
    seeg_labels = seeg_xyz.index.values
    region_centers = con.centres
    regions = con.region_labels
    num_regions = len(regions)
    
    # initialize object to assist in moving seeg contacts
    movecontact = tvbsim.util.MoveContacts(seeg_labels, seeg_xyz, 
                                       regions, region_centers, True)
    

    ezindex = iregion
    randpz = np.random.randint(0, len(regions), size=pznum)
    # ensure ez and pz never overlap
    while randpz in ezindex:
        randpz = np.random.randint(0, len(regions), size=pznum)

    if eznum <= 1:
        ezregion = list(regions[iregion])
    else:
        ezregion = regions[iregion]
    if pznum >= 1:
        pzregion = list(regions[randpz])
    elif pznum == 0:
        pzregion = []
    else:
        print >> sys.stderr, "Not implemented pz num >= 1 yet"
        raise
    
    ezindices = movecontact.getindexofregion(ezregion)
    pzindices = movecontact.getindexofregion(pzregion)
    
    ########## MOVE INDICES
    # move electrodes onto ez indices
    elecmovedindices = []
    for ezindex in ezindices:
        print "Moving onto current ez index: ", ezindex, " at ", regions[ezindex]
         # find the closest contact index and distance
        seeg_index, distance = movecontact.findclosestcontact(ezindex, elecmovedindices)

        # get the modified seeg xyz and gain matrix
        if movedistance != 0:
            modseeg, electrodeindices = movecontact.movecontactto(ezindex, seeg_index, distance)
        else:
            print "\n\nmoved contact onto ez exactly!!!!\n\n"
            modseeg, electrodeindices = movecontact.movecontact(ezindex, seeg_index)
        elecmovedindices.append(electrodeindices)
    
    # use subcortical structures!
    use_subcort = 1
    verts, normals, areas, regmap = tvbsim.util.read_surf(project_dir, use_subcort)
    modgain = tvbsim.util.gain_matrix_inv_square(verts, areas,
                        regmap, len(regions), movecontact.seeg_xyz)
    print "modified gain matrix the TVB way!"


    ezindices = movecontact.getindexofregion(ezregion)
    pzindices = movecontact.getindexofregion(pzregion)
    if not isinstance(ezindices, list):
        ezindices = np.array([ezindices])
    if not isinstance(pzindices, list):
        pzindices = np.array([pzindices])

    filename = os.path.join(outputdir, patient+'_sim_nez'+str(len(ezregion))+\
                                        '_npz'+str(len(pzregion))+ '_' + str(iregion) + '.npz')

    sys.stdout.write("\nProject directory for meta data is : " + project_dir)
    sys.stdout.write("\nFile to be saved is: " + filename)
    
    ####################### 2. Neural Mass Model @ Nodes ######################
    epileptors = tvbsim.initializers.models.initepileptor(epileptor_r, epiks, epitt, epitau, x0norm, \
                              x0ez, x0pz, ezindices, pzindices, num_regions)    
    ####################### 3. Integrator for Models ##########################
    heunint = tvbsim.initializers.integrators.initintegrator(heun_ts, noise_cov, noiseon=True)
    
    ################## 4. Difference Coupling Between Nodes ###################
    coupl = tvbsim.initializers.coupling.initcoupling(a=1.)
    
    ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #############
    monitors = tvbsim.initializers.monitors.initmonitors(period, sensorsfile, gainmatfile, varindex)
    
    sys.stdout.write("\nmoving contacts for " + patient)
    # modify the config of the monitors
    monitors[1].sensors.locations = modseeg
    monitors[1].gain = modgain

    # get initial conditions and then setup entire simulation configuration
    initcond = initconditions(x0norm, num_regions)
    sim, configs = setupconfig(epileptors, con, coupl, heunint, monitors, initcond)
    times, epilepts, seegts = runsim(sim, sim_length)

    postprocessor = tvbsim.util.PostProcess(epilepts, seegts, times)
    ######################## POST PROCESSING #################################
    # post process by cutting off first 5 seconds of simulation
    # for now, don't, since intiial conditions
    times, epits, seegts, zts = postprocessor.postprocts(samplerate)
    
    seizonsets = []
    seizoffsets = []
    try:
        # get the onsettimes and offsettimes for ez/pz indices
        onsettimes, offsettimes = getonsetsoffsets(zts, np.array(ezindices), np.array(pzindices))

        print("\nseizure onsets:", onsettimes)
        print("seizure offsets:", offsettimes)
        minsize = np.min((len(onsettimes),len(offsettimes)))

        # get the actual seizure times and offsets
        seizonsets, seizoffsets = getseiztimes(onsettimes, offsettimes)

        print(seizonsets, seizoffsets)
    except:
        print("try again")

    lowcut = 0.1
    highcut = 499.
    fs = 1000.
    x = seegts
    # y = butter_highpass_filter(x, lowcut, fs, order=4)
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=4)
    
    ######################## SAVING ALL DATA #################################
    regions = configs.connectivity.region_labels

    # Save files
    meta = {
        'x0ez':x0ez,
        'x0pz':x0pz,
        'x0norm':x0norm,
        'regions': regions,
        'regions_centers': configs.connectivity.centres,
        'seeg_contacts': configs.monitors[1].sensors.labels,
        'seeg_xyz': configs.monitors[1].sensors.locations,
        'ez': regions[ezindices],
        'pz': regions[pzindices],
        'ezindices': ezindices,
        'pzindices': pzindices,
        'onsettimes':seizonsets,
        'offsettimes':seizoffsets,
        'patient':patient,
    }

    # save tseries
    np.savez_compressed(filename, epits=epits, seegts=y, \
             times=times, zts=zts, metadata=meta)

if __name__ == '__main__':
    # patients = ['id001_ac', 'id002_cj', 'id014_rb']
    patient = str(sys.argv[1]).lower()
    eznum = int(sys.argv[2])
    pznum = int(sys.argv[3])
    metadatadir = str(sys.argv[4])
    outputdatadir = str(sys.argv[5])
    iregion = int(sys.argv[6])

    sys.stdout.write('Running cluster simulation...\n')
    runclustersim(patient,eznum,pznum,metadatadir,outputdatadir,iregion)
