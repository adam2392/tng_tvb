import sys
sys.path.append('./_tvblibrary/')
sys.path.append('./_tvbdata/')

from tvb.simulator.lab import *
import os
import time
import numpy as np
import pandas as pd 
# import scipy.io
import time
import timeit

# downloaded library for peak detection in z time series
import peakdetect
import scipy.signal as sig
import scipy.spatial.distance as dists
  
import tvbsim

def initconditions(x0norm, num_regions):
    epileptor_equil = epileptor_equil = models.Epileptor()
    epileptor_equil.x0 = x0norm
    init_cond = tvbsim.initialConditions.get_equilibrium(epileptor_equil, np.array([0.0, 0.0, 3.0, -1.0, 1.0, 0.0]))
    init_cond_reshaped = np.repeat(init_cond, num_regions).reshape((1, len(init_cond), num_regions, 1))

    return init_cond_reshaped

def setupconfig(model, con, coupl, integrator, monitors, init_cond=None):
    ############## 6. Initialize Simulator #############
    # initialize simulator object
    sim = simulator.Simulator(model=model,
                              initial_conditions=init_cond,
                              connectivity=con,
                              coupling=coupl,
                              integrator=integrator,
                              monitors=monitors)
    configs = sim.configure()
    return sim, configs

def runsim(sim, sim_length):
    # (epitimes, epilepts), (seegtimes, seegts) = sim.run(simulation_length=sim_length)

    # assuming model and seeg sampled at exactly same moment without 
    # conduction delay for now
    (times, epilepts), (_, seegts) = sim.run(simulation_length=sim_length)
    return times, epilepts, seegts

if __name__ == '__main__':
    # extract passed in variable
    patient = str(sys.argv[1]).lower() # ex: id001_ac
    MOVECONTACT = int(sys.argv[2])

    if not patient:
        patient='id002_cj'
    if not MOVECONTACT:
        MOVECONTACT = 1

    outputdir = os.path.join('/Users/adam2392/Documents/pydata/tvbforwardsim/', patient)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    root_dir = os.path.join('/Users/adam2392/Documents/pydata/')
    project_dir = os.path.join(root_dir, "metadata/",patient)
    tvbsim.util.renamefiles(patient, project_dir)
   
    print "Project directory for meta data is : ", project_dir
    print "File to be saved is: ", filename

    ####### Initialize files needed to run tvb simulation
    sensorsfile = os.path.join(project_dir, "seeg.txt")
    gainmatfile = os.path.join(project_dir, "gain_inv-square.txt")
    confile = os.path.join(project_dir, "connectivity.zip")
    
    use_subcort = 1
    verts, normals, areas, regmap = tvbsim.util.read_surf(project_dir, use_subcort)

    ########################### Simulation Parameters ############################
    samplerate = 1000 # Hz | 1000 = 1 second
    sim_length = 240*samplerate
    period = 1

    eznum = 1
    pznum = 2

    ########################### Epileptor Parameters ############################
    divisor = 4
    # intialized hard coded parameters
    epileptor_r = 0.0002 / divisor   # Temporal scaling in the third state variable
    epiks = -0.5                 # Permittivity coupling, fast to slow time scale
    epitt = 0.05               # time scale of simulation
    epitau = 10                # Temporal scaling coefficient in fifth st var
    x0norm= -2.5             # x0c value = -2.05
    x0ez= -1.6
    x0pz= -2.0

    # depends on epileptor variables of interest: it is where the x2-y2 var is
    varindex = [1]

    ########################### Integrator Parameters ############################
    # parameters for heun-stochastic integrator
    heun_ts = 0.05
    noise_cov = np.array([0.001, 0.001, 0.,\
                    0.0001, 0.0001, 0.])
  

    ####################### 1. Structural Connectivity ########################
    con = tvbsim.initializers.connectivity.initconn(confile)
    
    # extract the seeg_xyz coords and the region centers
    seeg_xyz = tvbsim.util.extractseegxyz(sensorsfile)
    seeg_labels = seeg_xyz.index.values
    region_centers = con.centres
    regions = con.region_labels
    num_regions = len(regions)
    
  
    randez = np.random.randint(0, len(regions), size=eznum)
    randpz = np.random.randint(0, len(regions), size=pznum)
    if eznum <= 1:
        ezregion = list(regions[randez])
    else:
        ezregion = regions[randez]
    if pznum <= 1:
        pzregion = list(regions[randpz])
    elif pznum > 1:
        pzregion = regions[randpz]
    elif pznum == 0:
        pzregion = []
    else:
        print >> sys.stderr, "Not implemented pz num >= 1 yet"
        raise
    
    # initialize object to assist in moving seeg contacts
    movecontact = tvbsim.util.MoveContacts(seeg_labels, seeg_xyz, 
                                       regions, region_centers, True)
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
        print "modified gain matrix the TVB way!"
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

    ezindices = movecontact.getindexofregion(ezregion)
    pzindices = movecontact.getindexofregion(pzregion)

    filename = os.path.join(outputdir, patient+'_sim_nez'+str(len(ezregion))+\
                                        '_npz'+str(len(pzregion))+'.npz')
    print "Project directory for meta data is : ", project_dir
    print "File to be saved is: ", filename

    ####################### 2. Neural Mass Model @ Nodes ######################
    epileptors = tvbsim.initializers.models.initepileptor(epileptor_r, epiks, epitt, epitau, x0norm, \
                              x0ez, x0pz, ezindices, pzindices, num_regions)    
    ####################### 3. Integrator for Models ##########################
    heunint = tvbsim.initializers.integrators.initintegrator(heun_ts, noise_cov)
    ################## 4. Difference Coupling Between Nodes ###################
    coupl = tvbsim.initializers.coupling.initcoupling(a=1.)
    ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #############
    monitors = tvbsim.initializers.monitors.initmonitors(period, sensorsfile, gainmatfile, varindex)
    
    if MOVECONTACT:
        # modify the config of the monitors
        monitors[1].sensors.locations = movecontact.seeg_xyz
        monitors[1].sensors.locations = modseeg
        monitors[1].gain = modgain


    ############################ RUN SIM ####################################
    # get initial conditions and then setup entire simulation configuration
    initcond = initconditions(x0norm, num_regions)
    sim, configs = setupconfig(model=epileptors, con=con, coupl=coupl, 
                    integrator=heunint, monitors=monitors, initcond=initcond)


    times, epilepts, seegts = runsim(sim=sim, sim_length=sim_length)
    postprocessor = tvbsim.util.PostProcess(epilepts, seegts, times)
    
    ######################## POST PROCESSING #################################
    # post process by cutting off first 5 seconds of simulation
    # for now, don't, since intiial conditions
    times, epits, seegts, zts = postprocessor.postprocts(samplerate)

    # get the onset, offset times
    onsettimes = None
    offsettimes = None
    try:
        onsettimes, offsettimes = PostProcess.findonsetoffset(zts[ezindices, :].squeeze(), delta=0.2/divisor)
    except:
        print "Either no ez was simulated, or need to correct delta in finding onset/offset."

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
