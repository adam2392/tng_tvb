import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
from tvb.simulator.lab import *

import numpy as np

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