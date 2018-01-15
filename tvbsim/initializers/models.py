import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
from tvb.simulator.lab import *

import numpy as np

def initepileptor(r, ks, tt, tau, x0norm, x0ez, x0pz,
                    ezindices, pzindices, num_regions):
    '''
    State variables for the Epileptor model:

    Repeated here for redundancy:

    x1 = first
    y1 = second
    z = third
    x2 = fourth
    y2 = fifth
    '''

    ####################### 2. Neural Mass Model @ Nodes ######################
    epileptors = models.Epileptor(Ks=ks, r=r, tau=tau, 
                tt=tt, variables_of_interest=['z', 'x2-x1'])

    # set x0 values (degree of epileptogenicity) for entire model
    epileptors.x0 = np.ones(num_regions) * x0norm
    # set ez region
    epileptors.x0[ezindices] = x0ez
    # set pz regions
    epileptors.x0[pzindices] = x0pz

    # Constrain the epileptor's state variable ranges. Consider getting rid of.
    # epileptors.state_variable_range['x1'] = np.r_[-0.5, 0.1]
    # epileptors.state_variable_range['z'] = np.r_[3.5,3.7]
    # epileptors.state_variable_range['y1'] = np.r_[-0.1,1]
    # epileptors.state_variable_range['x2'] = np.r_[-2.,0.]
    # epileptors.state_variable_range['y2'] = np.r_[0.,2.]
    # epileptors.state_variable_range['g'] = np.r_[-1.,1.]

    return epileptors