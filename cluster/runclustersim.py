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

import multiprocessing as mp
from contextlib import closing

def runclustersim(patient,eznum=1,pznum=0):
    print(patient)
    # eznum = 1
    # pznum = 0
    MOVECONTACT = 1

    # 1000 = 1 second
    samplerate = 1000 # Hz
    sim_length = 240*samplerate    
    period = 1

    ######### Region Parameters ##########
#     ezregion = ['ctx-lh-bankssts', 'ctx-lh-cuneus']
#     ezregion = ['ctx-lh-bankssts']
#     ezregion = []
#     pzregion = ['ctx-lh-cuneus']
#     pzregion = []

    ######### Epileptor Parameters ##########
    # intialized hard coded parameters
    epileptor_r = 0.0002       # Temporal scaling in the third state variable
    epiks = -15               # Permittivity coupling, fast to slow time scale
    epitt = 0.05               # time scale of simulation
    epitau = 10                # Temporal scaling coefficient in fifth st var
    # x0c value = -2.05
    x0norm=-2.5
    x0ez=-1.6
    x0pz=-2.04

    # depends on epileptor variables of interest: it is where the x2-y2 var is
    varindex = [1]

    ######### Integrator Parameters ##########
    # parameters for heun-stochastic integrator
    heun_ts = 0.05
    noise_cov = np.array([0.001, 0.001, 0.,\
                    0.0001, 0.0001, 0.])

    root_dir = os.path.join('/home/adamli/')
    project_dir = os.path.join(root_dir, "metadata/",patient)
    outputdir = os.path.join('/home/adamli/data/tvbforwardsim/', patient)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        
    tvbsim.util.renamefiles(patient, project_dir)

    ####### Initialize files needed to 
    # convert seeg.xyz to seeg.txt file
    sensorsfile = os.path.join(project_dir, "seeg.txt")
    # convert gain_inv-square.mat file into gain_inv-square.txt file
    gainmatfile = os.path.join(project_dir, "gain_inv-square.txt")
    confile = os.path.join(project_dir, "connectivity.zip")

    ####################### 1. Structural Connectivity ########################
    con = initconn(confile)

    # extract the seeg_xyz coords and the region centers
    seeg_xyz = tvbsim.util.extractseegxyz(sensorsfile)
    seeg_labels = seeg_xyz.index.values
    region_centers = con.centres
    regions = con.region_labels
    num_regions = len(regions)

    # initialize object to assist in moving seeg contacts
    movecontact = tvbsim.util.MoveContacts(seeg_labels, seeg_xyz, 
                                       regions, region_centers, True)

    
    randez = np.random.randint(0, len(regions), size=eznum)
    if eznum <= 1:
        ezregion = list(regions[randez])
    else:
        randez = np.random.randint(0, len(regions), size=pznum)
        ezregion = regions[randez]
    if pznum == 1:
        pzregion = list(regions[randpz])
    elif pznum == 0:
        pzregion = []
    else:
        print >> sys.stderr, "Not implemented pz num >= 1 yet"
        break
    
    
    if MOVECONTACT:
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
            modseeg, electrodeindices = movecontact.movecontact(ezindex, seeg_index)
            elecmovedindices.append(electrodeindices)
        
        # use subcortical structures!
        use_subcort = 1
        verts, normals, areas, regmap = tvbsim.util.read_surf(project_dir, use_subcort)
        modgain = tvbsim.util.gain_matrix_inv_square(verts, areas,
                            regmap, len(regions), movecontact.seeg_xyz)
        print("modified gain matrix the TVB way!")
        # simplest gain matrix will result in an error due to moving electrode 
        # directly onto an ez region, or any region for that matter
#         modgain = movecontact.simplest_gain_matrix(movecontact.seeg_xyz)
    elif MOVECONTACT == 0:
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
#         ezregion = np.append(ezregion,'ctx-lh-cuneus')
    ezindices = movecontact.getindexofregion(ezregion)
    pzindices = movecontact.getindexofregion(pzregion)
        
    filename = os.path.join(outputdir, patient+'_sim_nez'+str(len(ezregion))+\
                                        '_npz'+str(len(pzregion))+'.npz')
    print("Project directory for meta data is : ", project_dir)
    print("File to be saved is: ", filename)
    
    ####################### 2. Neural Mass Model @ Nodes ######################
    epileptors = initepileptor(epileptor_r, epiks, epitt, epitau, x0norm, \
                              x0ez, x0pz, ezindices, pzindices, num_regions)    
    ####################### 3. Integrator for Models ##########################
    heunint = initintegrator(heun_ts, noise_cov)
    ################## 4. Difference Coupling Between Nodes ###################
    coupl = initcoupling(a=1.)
    ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #############
    monitors = initmonitors(period, sensorsfile, gainmatfile, varindex)

    if MOVECONTACT:
        print "moving contacts for ", patient
        # modify the config of the monitors
        monitors[1].sensors.locations = movecontact.seeg_xyz
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

    # get the onset, offset times
    onsettimes = None
    offsettimes = None
    try:
        onsettimes, offsettimes = postprocessor.findonsetoffset(zts[ezindices, :].squeeze(), delta=0.2)
    except:
        print("Still not working...")

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
        'onsettimes':onsettimes,
        'offsettimes':offsettimes,
        'patient':patient,
    }

    # save tseries
    np.savez_compressed(filename, epits=epits, seegts=seegts, \
             times=times, zts=zts, metadata=meta)

if __name__ == '__main__':
    # patients = ['id001_ac', 'id002_cj', 'id014_rb']
    patient = str(sys.argv[1]).lower()
    eznum = int(sys.argv[2])
    pznum = int(sys.argv[3])

    runclustersim(patient,eznum,pznum)
