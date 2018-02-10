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

def sample_ezpz_regions(regions, eznum, pznum):
    randez = np.random.randint(0, len(regions), size=eznum)
    randpz = np.random.randint(0, len(regions), size=pznum)

    randez = np.where(regions == 'ctx-rh-middletemporal')[0]
    randpz = np.where(regions == 'ctx-rh-lateraloccipital')[0]

    # ensure ez and pz never overlap
    while randpz in randez:
        randpz = np.random.randint(0, len(regions), size=pznum)

    if eznum == 0:
        ezregion = []
    elif eznum <= 1:
        ezregion = list(regions[randez])
    else:
        ezregion = regions[randez]
    if pznum >= 1:
        pzregion = list(regions[randpz])
    elif pznum == 0:
        pzregion = []

    return ezregion, pzregion

def runclustersim(patient,eznum=1,pznum=0,metadatadir=None,outputdatadir=None,iregion=-1):
    sys.stdout.write(patient)
    ###### SIMULATION LENGTH AND SAMPLING ######
    # 1000 = 1 second
    samplerate = 1000 # Hz
    sim_length = 180*samplerate    
    period = 1
    movedistance = 0

    ######### Epileptor Parameters ##########
    # intialized hard coded parameters
    epileptor_r = 0.0002#/1.5   # Temporal scaling in the third state variable
    epiks = -2                  # Permittivity coupling, fast to slow time scale
    epitt = 0.1               # time scale of simulation
    epitau = 10                # Temporal scaling coefficient in fifth st var
    
    # x0c value = -2.05
    x0norm=-2.4
    x0ez=-1.7
    x0pz=-2.0

    # depends on epileptor variables of interest: it is where the x2-y2 var is
    varindex = [1]

    ######### Integrator Parameters ##########
    # parameters for heun-stochastic integrator
    heun_ts = 0.05
    noise_cov = np.array([0.001, 0.001, 0.,\
                    0.0001, 0.0001, 0.])/4

    sim_params = {'r': epileptor_r,
                'epiks': epiks,
                'epitt': epitt,
                'epitau': epitau,
                'noise': noise_cov}

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
    
    # get a sampled ez, pz region
    ezregion, pzregion = sample_ezpz_regions(regions, eznum, pznum)
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


    ezindices = movecontact.getindexofregion(ezregion)
    pzindices = movecontact.getindexofregion(pzregion)
    if not isinstance(ezindices, list):
        ezindices = np.array([ezindices])
    if not isinstance(pzindices, list):
        pzindices = np.array([pzindices])

    # filename = os.path.join(outputdir, patient+'_sim_nez'+str(len(ezregion))+\
    #                                     '_npz'+str(len(pzregion))+ '_' + str(iregion) + '.npz')
    filename = os.path.join(outputdir, patient+'_sim_nez'+str(len(ezregion))+\
                                        '_npz'+str(len(pzregion)) + '.npz')


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
    if eznum > 0:
        monitors[1].sensors.locations = modseeg
        monitors[1].gain = modgain

    # get initial conditions and then setup entire simulation configuration
    initcond = initconditions(x0norm, num_regions)
    initcond = None

    sim, configs = setupconfig(epileptors, con, coupl, heunint, monitors, initcond)
    times, epilepts, seegts = runsim(sim, sim_length)

    postprocessor = tvbsim.util.PostProcess(epilepts, seegts, times)
    ######################## POST PROCESSING #################################
    # post process by cutting off first 5 seconds of simulation
    # for now, don't, since intiial conditions
    times, epits, seegts, zts = postprocessor.postprocts(samplerate)
    
    # determine where norm, pz and ez are
    ezindices = np.where(config.model.x0 == x0ez)[0]
    pzindices = np.where(config.model.x0 == x0pz)[0]
    settimes = postprocessor.getonsetsoffsets(zts, ezindices, pzindices, delta=0.2/5)
    # get the actual seizure times and offsets
    seizonsets, seizoffsets = postprocessor.getseiztimes(settimes)

    lowcut = 0.1
    highcut = 499.
    fs = 1000.
    x = seegts
    # y = butter_highpass_filter(x, lowcut, fs, order=4)
    y = tvbsim.util.butter_bandpass_filter(x, lowcut, highcut, fs, order=4)
    
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
        'simparams': sim_params
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
    # iregion = int(sys.argv[6])

    sys.stdout.write('Running cluster simulation...\n')
    runclustersim(patient,eznum,pznum,metadatadir,outputdatadir)
