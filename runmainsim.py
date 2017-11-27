import sys
sys.path.append('./_tvblibrary/')
sys.path.append('./_tvbdata/')

from tvb.simulator.lab import *
import os
import time
import numpy as np
import pandas as pd 
import scipy.io
import time
import timeit

# downloaded library for peak detection in z time series
import peakdetect
import scipy.signal as sig
import scipy.spatial.distance as dists
  
import tvbsim

def initconn(confile):
    ####################### 1. Structural Connectivity ########################
    con = connectivity.Connectivity.from_file(confile)
    # set connectivity speed to instantaneous
    con.speed = np.inf
    # normalize weights
    con.weights = con.weights/np.max(con.weights)

    # To avoid adding analytical gain matrix for subcortical sources
    con.cortical[:] = True     

    return con

def initepileptor(r, ks, tt, tau, x0norm, x0ez, x0pz, ezindices, pzindices, num_regions):
    ####################### 2. Neural Mass Model @ Nodes ######################
    epileptors = models.Epileptor(Ks=ks, r=r, tau=tau, tt=tt, variables_of_interest=['z', 'x2-x1'])

    # set x0 values (degree of epileptogenicity) for entire model
    epileptors.x0 = np.ones(num_regions) * x0norm
    # set ez region
    epileptors.x0[ezindices] = x0ez
    # set pz regions
    epileptors.x0[pzindices] = x0pz

    epileptors.state_variable_range['x1'] = np.r_[-0.5, 0.1]
    epileptors.state_variable_range['z'] = np.r_[3.5,3.7]
    epileptors.state_variable_range['y1'] = np.r_[-0.1,1]
    epileptors.state_variable_range['x2'] = np.r_[-2.,0.]
    epileptors.state_variable_range['y2'] = np.r_[0.,2.]
    epileptors.state_variable_range['g'] = np.r_[-1.,1.]

    return epileptors

def initintegrator(heun_ts, noise_cov):
    ####################### 3. Integrator for Models ##########################
    # define cov noise for the stochastic heun integrato
    hiss = noise.Additive(nsig=noise_cov)
    heunint = integrators.HeunStochastic(dt=heun_ts, noise=hiss)
    return heunint

def initcoupling(a):
    ################## 4. Difference Coupling Between Nodes ###################
    # define a simple difference coupling
    coupl = coupling.Difference(a=1.)
    return coupl

def initmonitors(period, seegfile, gainmatfile, varindex):
    ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #############
    mon_tavg = monitors.TemporalAverage(period=period) # monitor model
    mon_SEEG = monitors.iEEG.from_file(sensors_fname=seegfile,
                        projection_fname=gainmatfile,
                        period=period,
                        variables_of_interest=[1]
                    )

    return [mon_tavg, mon_SEEG]

def initconditions(x0norm, num_regions):
    epileptor_equil = epileptor_equil = models.Epileptor()
    epileptor_equil.x0 = x0norm
    init_cond = tvbsim.initialConditions.get_equilibrium(epileptor_equil, np.array([0.0, 0.0, 3.0, -1.0, 1.0, 0.0]))
    init_cond_reshaped = np.repeat(init_cond, num_regions).reshape((1, len(init_cond), num_regions, 1))

    return init_cond_reshaped

def setupconfig(epileptors, con, coupl, heunint, monitors, init_cond_reshaped=None):
    ############## 6. Initialize Simulator #############
    # initialize simulator object
    sim = simulator.Simulator(model=epileptors,
                              initial_conditions=init_cond_reshaped,
                              connectivity=con,
                              coupling=coupl,
                              integrator=heunint,
                              monitors=monitors)
    configs = sim.configure()
    return sim, configs

def initentiremodel(confile, seegfile, gainmatfile, period, ezindices, pzindices):
    # x0c value = -2.05
    x0norm=-2.5
    x0ez=-1.6
    x0pz=-2.05
    
    # intialized hard coded parameters
    epileptor_r = 0.0002       # Temporal scaling in the third state variable
    epiks = -0.5                 # Permittivity coupling, fast to slow time scale
    epitt = 0.05               # time scale of simulation
    epitau = 10                # Temporal scaling coefficient in fifth st var

    # depends on epileptor variables of interest: it is where the x2-y2 var is
    varindex = [1]
    
    # parameters for heun-stochastic integrator
    heun_ts = 0.05
    noise_cov = np.array([0.001, 0.001, 0.,\
                    0.0001, 0.0001, 0.])
    
    # use new coord/gain mat if avail
    if modseeg:
        mon_SEEG.sensors.locations = modseeg
        mon_SEEG.gain = modgain
        print "modified gain and chan xyz"

    ######## 1. Connectivity
    conn = initconn(confile)
    ######## 2. Model
    epileptors =initepileptor(epileptor_r, epiks, epitt, epitau, x0norm, \
                              x0ez, x0pz, ezindices, pzindices, num_regions)    
    ####################### 3. Integrator for Models ##########################
    heunint = initintegrator(heun_ts, noise_cov)
    ################## 4. Difference Coupling Between Nodes ###################
    coupl = initcoupling(a=1.)
    ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #############
    monitors = initmonitors(period, seegfile, gainmatfile, varindex)

    # get initial conditions and then setup entire simulation configuration
    initcond = initconditions(x0norm, num_regions)
    sim, configs = setupconfig(epileptors, con, coupl, heunint, monitors, initcond)
    
    return sim, configs

def runsim(sim, sim_length):
    # (epitimes, epilepts), (seegtimes, seegts) = sim.run(simulation_length=sim_length)

    # assuming model and seeg sampled at exactly same moment without 
    # conduction delay for now
    (times, epilepts), (_, seegts) = sim.run(simulation_length=sim_length)
    return times, epilepts, seegts


def getindexofregion(regions, ezregion=[], pzregion=[]):
    sorter = np.argsort(regions)
    ezindices = sorter[np.searchsorted(regions, ezregion, sorter=sorter)]
    pzindices = sorter[np.searchsorted(regions, pzregion, sorter=sorter)]

    return ezindices, pzindices

def postprocts(epits, seegts, times, samplerate=1000):
    # reject certain 5 seconds of simulation
    secstoreject = 7
    sampstoreject = secstoreject * samplerate

    # get the time series processed and squeezed that we want to save
    new_times = times[sampstoreject:]
    new_epits = epits[sampstoreject:, 1, :, :].squeeze().T
    new_zts = epits[sampstoreject:, 0, :, :].squeeze().T
    new_seegts = seegts[sampstoreject:, :, :, :].squeeze().T

    # don't reject any time period
    new_times = times
    new_epits = epits[:, 1, :, :].squeeze().T
    new_zts = epits[:, 0, :, :].squeeze().T
    new_seegts = seegts[:,:, :, :].squeeze().T

    return new_times, new_epits, new_seegts, new_zts

# assuming onset is the first bifurcation and then every other one is onsets
# every other bifurcation after the first one is the offset
def findonsetoffset(zts):
    maxpeaks, minpeaks = peakdetect.peakdetect(zts)
    
    # get every other peaks
    onsettime, _ = zip(*maxpeaks)
    offsettime, _ = zip(*minpeaks)
    
    return onsettime, offsettime

if __name__ == '__main__':
    patient='id002_cj'
    # 1000 = 1 second
    samplerate = 1000 # Hz
    sim_length = 5*samplerate

    ezregion = ['ctx-lh-cuneus']
    pzregion = []

    x0norm=-2.3
    x0ez=-1.8
    x0pz=2.05

    root_dir = os.getcwd()
    project_dir = os.path.join(root_dir, "metadata/"+patient)
    print "Project directory for meta data is : ", project_dir

    outputdir = os.path.join(root_dir, 'output')
    filename = os.path.join(outputdir, patient+'_sim_nez'+str(len(ezregion))+\
                                    '_npz'+str(len(pzregion))+'.npz')

    confile = os.path.join(project_dir, "connectivity.zip")

    # convert seeg.xyz to seeg.txt file
    sensorsfile = os.path.join(project_dir, "seeg.xyz")
    newsensorsfile = os.path.join(project_dir, "seeg.txt")
    try:
        os.rename(sensorsfile, newsensorsfile)
    except:
        print "Already renamed seeg.xyz possibly!"

    # convert gain_inv-square.mat file into gain_inv-square.txt file
    gainmatfile = os.path.join(project_dir, "gain_inv-square.mat")
    newgainmatfile = os.path.join(project_dir, "gain_inv-square.txt")
    try:
        os.rename(gainmatfile, newgainmatfile)
    except:
        print "Already renamed gain_inv-square.mat possibly!"

    ######################## SIM PROCESSING #################################
    sim, configs = initmodel(confile, newsensorsfile, newgainmatfile, ezregion=ezregion, pzregion=pzregion, x0norm=x0norm, x0ez=x0ez, x0pz=x0pz)
    times, epilepts, seegts = runsim(sim, sim_length)

    ######################## POST PROCESSING #################################
    regions = configs.connectivity.region_labels
    # post process by cutting off first 5 seconds of simulation
    times, epits, seegts, zts = postprocts(epilepts, seegts, times)

    ezindices, pzindices = getindexofregion(regions, ezregion, pzregion)
    # get the onset, offset times
    onsettimes = None
    offsettimes = None
    try:
        onsettimes, offsettimes = findonsetoffset(zts[:, ezindices].squeeze())
    except:
        print "Still not working..."

    ######################## SAVING ALL DATA #################################
    # Save files
    meta = {
        'x0ez':x0ez,
        'x0pz':x0pz,
        'x0norm':x0norm,
        'ez': regions[ezindices],
        'pz': regions[pzindices],
        'ezindices': ezindices,
        'pzindices': pzindices,
        'onsettimes':onsettimes,
        'offsettimes':offsettimes,
        'patient':patient
    }

    # save tseries
    np.savez(filename, epits=epits, seegts=seegts, \
             times=times, zts=zts, metadata=meta)
