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
from runmainsim import *
import tvbsim

'''
This script is only used for generating generic simulations.

Contacts stay in the same location as specified for this patient clinically.
Gain matrix stays the same. The only paramters that are worth changing are the ones
that influence the simulation itself (i.e. r, Ks, tau, etc.).
'''

def runclustersim(patient,metadatadir,outputdatadir,movedist=None,
                            x0ez=None,x0pz=None,iregion=0):
    sys.stdout.write(patient)
    if not os.path.exists(outputdatadir):
        os.makedirs(outputdatadir)
    project_dir = os.path.join(metadatadir, patient)
    outputdir = os.path.join(outputdatadir, patient)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    tvbsim.util.renamefiles(patient, project_dir)

    use_subcort = 1
    verts, normals, areas, regmap = tvbsim.util.read_surf(project_dir, use_subcort)

    samplerate = 1000 # Hz
    sim_length = 135*samplerate    
    period = 1

    ######### Epileptor Parameters ##########
    epileptor_r = 0.00035#/1.5   # Temporal scaling in the third state variable
    epiks = -2                  # Permittivity coupling, fast to slow time scale
    epitt = 0.06               # time scale of simulation
    epitau = 10                # Temporal scaling coefficient in fifth st var

    # x0c value = -2.05
    x0norm=-2.35
    x0ez=-1.85
    x0pz=-2.05
    # eznum = 1
    # pznum = 1

    ######### Integrator Parameters ##########
    # parameters for heun-stochastic integrator
    heun_ts = 0.05
    # noise on state_variables = ['x1', 'y1', 'z', 'x2', 'y2', 'g']
    noise_cov = np.array([0.001, 0.001, 0.,\
                          0.0001, 0.0001, 0.])
    sim_params = {'r': epileptor_r,
                  'epiks': epiks,
                  'epitt': epitt,
                  'epitau': epitau,
                  'noise': noise_cov}
    epi_params = {'r': epileptor_r,
                  'ks': epiks,
                  'tt': epitt,
                  'tau': epitau,
                  'x0norm': x0norm,
                  'x0pz': x0pz,
                  'x0ez': x0ez}
    heun_params = {'ts': heun_ts,
                   'noise': noise_cov}
    surf_params = dict()
    surf_params['verts'] = verts
    surf_params['normals'] = normals
    surf_params['areas'] = areas
    surf_params['regmap'] = regmap
            
    ####### Initialize files needed to 
    # convert seeg.xyz to seeg.txt file
    sensorsfile = os.path.join(project_dir, "seeg.txt")
    gainmatfile = os.path.join(project_dir, "gain_inv-square.txt")
    confile = os.path.join(project_dir, "connectivity.zip")

    if 'id008' in patient:
        # 008
        # ezregions = ['Right-Amygdala', 'Right-Hippocampus']
        # pzregions = ['ctx-rh-superiortemporal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal',
        #      'ctx-rh-medialorbitofrontal', 'ctx-rh-lateralorbitofrontal']

        ezregions = ['Right-Amygdala']
        pzregions = ['ctx-rh-superiortemporal']
    elif 'id013' in patient:
        # 013
        ezregions = ['ctx-rh-fusiform']
        # pzregions = ['ctx-rh-inferiortemporal','Right-Hippocampus','Right-Amygdala', 
        #       'ctx-rh-middletemporal','ctx-rh-entorhinal']

        pzregions = ['ctx-rh-inferiortemporal']
    elif 'id001' in patient:
        ezregions = ['ctx-rh-lateralorbitofrontal', 'ctx-rh-temporalpole']
        pzregions = ['ctx-rh-superiorfrontal', 'ctx-rh-rostralmiddlefrontal', 'ctx-lh-lateralorbitofrontal']

    # initialize file directory to save simulation
    if movedist >=0:
        filename = os.path.join(outputdir, 
            patient+'_sim_x0ez'+str(x0ez)+'_x0pz'+str(x0pz)+ '_dist' + str(movedist) + '.npz')
    else:
        filename = os.path.join(outputdir, 
            patient+'_sim_nez'+str(eznum)+'_npz'+str(pznum) + '.npz')
        filename = os.path.join(outputdir, 
                patient+'_sim_x0ez'+str(x0ez)+'_x0pz'+str(x0pz) + '.npz')

    sys.stdout.write("\nProject directory for meta data is : " + project_dir)
    sys.stdout.write("\nFile to be saved is: " + filename)


    # setup the simulation using params
    epileptors, con, coupl, heunint, monitor, initcond = tvbsim.initsim.initializesim(period, epi_params, heun_params, 
                                                        confile, sensorsfile, gainmatfile, 
                                                        eznum=None, pznum=None, ezregion=ezregions, pzregion=pzregions,
                                                        surf_params=surf_params, movedist=movedist)
    # set init condition to none....
    initcond = None
    
    sim, config = setupconfig(epileptors, con, coupl, heunint, monitor, initcond)
    times, epilepts, seegts = runsim(sim, sim_length)

    postprocessor = tvbsim.util.PostProcess(epilepts, seegts, times)
    ######################## POST PROCESSING #################################
    # post process by cutting off first 5 seconds of simulation
    # for now, don't, since intiial conditions
    times, epits, seegts, zts = postprocessor.postprocts(samplerate)
    
    regions = config.connectivity.region_labels
    # determine where norm, pz and ez are
    ezindices = np.where(config.model.x0 == x0ez)[0]
    pzindices = np.where(config.model.x0 == x0pz)[0]

    ezregions = regions[ezindices]
    if len(pzindices) > 0:
        pzregions = regions[pzindices]
        pzindices = np.array([])
    else:
        pzregions = None

    seizonsets = []
    seizoffsets = []
    allindices = np.append(ezindices, pzindices, axis=0).astype(int)
    print(allindices)
    settimes = postprocessor.getonsetsoffsets(zts, allindices, delta=0.2/5)
    # get the actual seizure times and offsets
    seizonsets, seizoffsets = postprocessor.getseiztimes(settimes)

    lowcut = 0.1
    highcut = 499.
    x = seegts
    # y = butter_highpass_filter(x, lowcut, fs, order=4)
    y = tvbsim.util.butter_bandpass_filter(x, lowcut, highcut, samplerate, order=4)

    ######################## SAVING ALL DATA #################################
    # Save files
    meta = {
        'x0ez':x0ez,
        'x0pz':x0pz,
        'x0norm':x0norm,
        'regions': regions,
        'regions_centers': config.connectivity.centres,
        'seeg_contacts': config.monitors[1].sensors.labels,
        'seeg_xyz': config.monitors[1].sensors.locations,
        'ez': ezregions,
        'pz': pzregions,
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
    # x0ez = float(sys.argv[2])
    # x0pz = float(sys.argv[3])
    metadatadir = str(sys.argv[2])
    outputdatadir = str(sys.argv[3])
    movedist = float(sys.argv[4])
    # iregion = int(sys.argv[6])

    sys.stdout.write('Running cluster simulation...\n')
    runclustersim(patient,metadatadir,outputdatadir,movedist)
