import numpy as np
from initializers import *
import util
import initialConditions

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

def initconditions(x0init, num_regions):
    epileptor_equil = models.initemptyepileptor()  
    epileptor_equil.x0 = x0init
    init_cond = initialConditions.get_equilibrium(epileptor_equil, np.array([0.0, 0.0, 3.0, -1.0, 1.0, 0.0]))
    init_cond_reshaped = np.repeat(init_cond, num_regions).reshape((1, len(init_cond), num_regions, 1))

    return init_cond_reshaped


def initializesim(period, epi_params, heun_params, 
                confile, sensorsfile, gainmatfile, 
                eznum=None, pznum=None, surf_params=None, movedist=-1, 
                ezregion=None, pzregion=None):
    # extract epileptor parameters
    epileptor_r = epi_params['r']
    epiks = epi_params['ks']
    epitt = epi_params['tt']
    epitau = epi_params['tau']
    x0norm = epi_params['x0norm']
    x0ez = epi_params['x0ez']
    x0pz = epi_params['x0pz']

    # extract heun params
    heun_ts = heun_params['ts']
    heun_noise = heun_params['noise']

    if surf_params:
        verts = surf_params['verts']
        normals = surf_params['normals']
        areas = surf_params['areas']
        regmap = surf_params['regmap']

    # depends on epileptor variables of interest: it is where the x2-y2 var is
    varindex = [1]

    ####################### 1. Structural Connectivity ########################
    con = connectivity.initconn(confile)
    
    # extract the seeg_xyz coords and the region centers
    seeg_xyz = util.extractseegxyz(sensorsfile)
    seeg_labels = seeg_xyz.index.values
    region_centers = con.centres
    regions = con.region_labels
    num_regions = len(regions)
    
    # initialize object to assist in moving seeg contacts
    movecontact = util.MoveContacts(seeg_labels, seeg_xyz, 
                                       regions, region_centers, True)
    
    # get a sampled ez, pz region
    # ezregion, pzregion = sample_ezpz_regions(regions, eznum, pznum)
    # if not ezregion:
    #     # ezregion = ['ctx-rh-fusiform']
    #     ezregion = ['ctx-rh-middletemporal']
    
    # if not pzregion:
    #     # pzregion = ['ctx-rh-inferiortemporal']
    #     pzregion = ['ctx-rh-lateraloccipital']

    ezregion = np.array(ezregion)
    pzregion = np.array(pzregion)
    ezindices = movecontact.getindexofregion(ezregion)
    pzindices = movecontact.getindexofregion(pzregion)

    # move the seeg contacts to a specified region
    if movedist >= 0:
        ########## MOVE INDICES
        # move electrodes onto ez indices
        elecmovedindices = []
        electrodeindices = []
        for ezindex in ezindices:
            print("Moving onto current ez index: ", ezindex, " at ", regions[ezindex])
             # find the closest contact index and distance
            seeg_index, distance = movecontact.findclosestcontact(ezindex, elecmovedindices)

            # get the modified seeg xyz and gain matrix
            modseeg, electrodeindices = movecontact.movecontactto(ezindex, seeg_index, movedist)
            elecmovedindices.append(electrodeindices)
        
        # get the modified gain
        modgain = util.gain_matrix_inv_square(verts, areas,
                            regmap, num_regions, modseeg)

        test = modseeg[seeg_index] - region_centers[ezindex,:]
        print('\n\n Am %s from ez region! Should match movedist!\n' % np.linalg.norm(test))
        print('This electrode has this many electrode indices on it %s' % electrodeindices)
    if not isinstance(ezindices, list):
        ezindices = np.array([ezindices])
    if not isinstance(pzindices, list):
        pzindices = np.array([pzindices])

    ####################### 2. Neural Mass Model @ Nodes ######################
    epileptors = models.initepileptor(epileptor_r, epiks, epitt, epitau, x0norm, \
                              x0ez, x0pz, ezindices, pzindices, num_regions) 


    # print('constraining epileptors state variable ranges')
    # epileptors.state_variable_range['x1'] = np.r_[-0.5, 0.1]
    # epileptors.state_variable_range['z'] =  np.r_[3.5,3.7]
    # epileptors.state_variable_range['y1'] = np.r_[-0.1,1]
    # epileptors.state_variable_range['x2'] = np.r_[-2.,0.]
    # epileptors.state_variable_range['y2'] = np.r_[0.,2.]
    # epileptors.state_variable_range['g'] = np.r_[-1.,1.]

    ####################### 3. Integrator for Models ##########################
    heunint = integrators.initintegrator(heun_ts, heun_noise, noiseon=True)
    
    ################## 4. Difference Coupling Between Nodes ###################
    coupl = coupling.initcoupling(a=1.)
    
    ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #############
    monitor = monitors.initmonitors(period, sensorsfile, gainmatfile, varindex)

    if movedist >= 0:
        monitor[1].sensors.locations = modseeg
        monitor[1].gain = modgain

    # get initial conditions and then setup entire simulation configuration
    x0init = -2.05
    # set x0 values (degree of epileptogenicity) for entire model
    # initarray = np.repeat(np.array([0.0, 0.0, 3.0, -1.0, 1.0, 0.0]), 84, axis=0)
    # initcond = initialConditions.get_equilibrium(epileptors, initarray)
    # initcond = initcond.reshape(1, 6, 84, 1)
    initcond = initconditions(x0init, num_regions)

    return epileptors, con, coupl, heunint, monitor, initcond