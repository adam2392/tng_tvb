import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
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
import tvbsim

# import re
# import sys
# sys.path.append('/home-1/ali39@jhu.edu/work/fragility_analysis/')
# import fragility.signalprocessing as sp

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

def setclosetsregion(movecontact, seeg_labels):
    ########## SET EZREGION BASED ON CLOSENESS
    # find the closest region-contact pair
    for idx, label in enumerate(seeg_labels):
        region_index, distance = movecontact.getregionsforcontacts(label)

        if idx == 0:
            mindist = distance
            minregion = region_index
            mincontact = label
        else:
            if distance < mindist:
                mindist = distance
                minregion = region_index
                mincontact = label
                
    ezregion = minregion

if __name__ == '__main__':
    # patients = ['id001_ac', 'id002_cj', 'id014_rb']
    patient = str(sys.argv[1]).lower()
    numez = int(sys.argv[2])
    numpz = int(sys.argv[3])
    metadatadir = str(sys.argv[4])
    outputdatadir = str(sys.argv[5])
    iproc = int(sys.argv[6]) # which processor currently being used

    sys.stdout.write('Running cluster simulation...\n')
    sys.stdout.write(patient)

    ###### SIMULATION LENGTH AND SAMPLING ######
    # 1000 = 1 second
    samplerate = 1000 # Hz
    sim_length = 180*samplerate    
    period = 1

    ######### Epileptor Parameters ##########
    # intialized hard coded parameters
    epileptor_r = 0.0002       # Temporal scaling in the third state variable
    epiks = -1               # Permittivity coupling, fast to slow time scale
    epitt = 0.025               # time scale of simulation
    epitau = 4                # Temporal scaling coefficient in fifth st var
    # x0c value = -2.05
    x0norm=-2.4
    x0ez=-1.7
    x0pz=-2.0

    # BANDPASS FILTER paramters for Postprocessing the SEEG
    lowcut = 0.1
    highcut = 499.
    fs = 1000.

    # depends on epileptor variables of interest: it is where the x2-y2 var is
    varindex = [1]

    ######### Integrator Parameters ##########
    # parameters for heun-stochastic integrator
    heun_ts = 0.05
    noise_cov = np.array([0.001, 0.001, 0.,\
                    0.0001, 0.0001, 0.])/2

    project_dir = os.path.join(metadatadir, patient)
    outputdir = os.path.join(outputdatadir, patient)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    tvbsim.util.renamefiles(patient, project_dir)

    ####### Initialize files needed to run tvb simulation
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

    eznum = numez
    pznum = numpz
    randez = np.random.randint(0, len(regions), size=eznum)
    randpz = np.random.randint(0, len(regions), size=pznum)

    # ensure ez and pz never overlap
    while randpz in randez:
        randpz = np.random.randint(0, len(regions), size=pznum)

    # get the ez/pz region labels
    if eznum <= 1:
        ezregion = list(regions[randez])
    else:
        ezregion = regions[randez]
    if pznum >= 1:
        pzregion = list(regions[randpz])
    elif pznum == 0:
        pzregion = []
    else:
        pzreiong = regions[randpz]
    
    ############################ ALGORITHM FOR MOVING CONTACTS 
    if MOVECONTACT:
        sys.stdout.write("\nmoving contacts for " + patient)

        ezindices = movecontact.getindexofregion(ezregion)
        pzindices = movecontact.getindexofregion(pzregion)
        
        ########## MOVE INDICES
        # move electrodes onto ez indices
        elecmovedindices = []
        for ezindex in ezindices:
            print("Moving onto current ez index: ", ezindex, " at ", regions[ezindex])
             # find the closest contact index and distance
            seeg_index, distance = movecontact.findclosestcontact(ezindex, elecmovedindices)

            # get the modified seeg xyz and gain matrix
            if movedistance != 0:
                modseeg, electrodeindices = movecontact.movecontactto(ezindex, seeg_index, distance)
            else:
                print("\n\nmoved contact onto ez exactly!!!!\n\n")
                modseeg, electrodeindices = movecontact.movecontact(ezindex, seeg_index)
            elecmovedindices.append(electrodeindices)
        
        # use subcortical structures!
        use_subcort = 1
        verts, normals, areas, regmap = tvbsim.util.read_surf(project_dir, use_subcort)
        modgain = tvbsim.util.gain_matrix_inv_square(verts, areas,
                            regmap, len(regions), movecontact.seeg_xyz)

        print("modified gain matrix the TVB way!")
    elif MOVECONTACT == 0:
        ezregion = setclosestregion(movecontact, seeg_labels)
    elif MOVECONTACT == -1:
        sys.stdout.write("\nNot changing anything about contacts\n")
  
    # get indices of ez region and pz regions if they changed
    ezindices = movecontact.getindexofregion(ezregion)
    pzindices = movecontact.getindexofregion(pzregion)
        
     ####################### SPECIFY OUTPUT DIRECTORY NAME ######################
    outputdir = os.path.join(outputdatadir, 
        'nez'+str(len(ezregion))+'_npz'+str(len(pzregion)) + '_v2' )
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    filename = os.path.join(outputdir, patient+'_sim_nez'+str(len(ezregion))+\
                                        '_npz'+str(len(pzregion))+'.npz')

    if not isinstance(ezindices, list):
        ezindices = np.array([ezindices])
    if not isinstance(pzindices, list):
        pzindices = np.array([pzindices])

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
    
    if MOVECONTACT:
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
        print(onsettimes, offsettimes)
    except:
        sys.stdout.write("try again!!!!!!!!!!!!!!!!!!!!")

    # y = butter_highpass_filter(x, lowcut, fs, order=4)
    seegts = butter_bandpass_filter(seegts, lowcut, highcut, fs, order=4)

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
    np.savez_compressed(filename, epits=epits, seegts=seegts, \
             times=times, zts=zts, metadata=meta)