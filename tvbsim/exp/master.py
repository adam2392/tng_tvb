import sys
import numpy as np
import pandas as pd
import os

from tvb.simulator.lab import *
import tvbsim
from tvbsim.exp.basetvbexp import TVBExp
from tvbsim.maintvbexp import MainTVBSim
from tvbsim.exp.utils.selectregion import Regions

class MasterExp(object):
    metadata = None

    def __init__(self, conn_params,
                        coupling_params,
                        model_params,
                        integrator_params,
                        monitor_params):
        self.conn_params = conn_params
        self.coupling_params = coupling_params
        self.model_params = model_params
        self.integrator_params = integrator_params
        self.monitor_params = monitor_params

        self.tvbexp = MainTVBSim()
    def setupsim(self):
        # set up connectivity
        self.tvbexp.loadconn(**self.conn_params)

        # load coupling
        self.tvbexp.loadcoupling(**self.coupling_params)

        # set up model
        self.tvbexp.loadmodel(**self.model_params)

        # set up integrator
        self.tvbexp.loadintegrator(**self.integrator_params)

        # set up monitors
        self.tvbexp.loadmonitors(**self.monitor_params)

        self.tvbexp.setupsim()
        self.allindices = np.hstack((self.tvbexp.ezind, self.tvbexp.pzind)).astype(int) 

    def runsim(self, sim_length):
        times, simvars, seegts = self.tvbexp.mainsim(sim_length=sim_length)
        return times, simvars, seegts

    def get_metadata(self):
        self.metadata = {
                'conn_params': self.conn_params,
                'coupling_params': self.coupling_params,
                'model_params': self.model_params,
                'integrator_params': self.integrator_params,
                'monitor_params': self.monitor_params,
        }
        return self.metadata
    
    def shuffleweights(self):
        # shuffle within patients
        randweights = self.maintvbexp.randshuffleweights(self.conn.weights)
        self.conn.weights = randweights
        randpat = None
        return conn, randpat

    def shufflepatients(self):   
        # shuffle across patients
        randpat = self.maintvbexp.randshufflepats(allpats, patient)   
        shuffled_connfile = os.path.join(self.root_dir, randpat, 'tvb', 'connectivity.zip')
        if not os.path.exists(shuffled_connfile):
            shuffled_connfile = os.path.join(self.root_dir, randpat, 'tvb', 'connectivity.dk.zip')
        conn = connectivity.Connectivity.from_file(shuffled_connfile)
        return conn, randpat

    def select_ez_outside(self, numsamps):
        # region selector for out of clinical EZ simulations
        epsilon = 60 # the mm radius for each region to exclude other regions
        regionselector = Regions(self.conn.region_labels, self.conn.centres, epsilon)
        # the set of regions that are outside what clinicians labeled EZ
        outside_set = regionselector.generate_outsideset(self.clinezregions)
        # sample it for a list of EZ regions
        osr_list = regionselector.sample_outsideset(outside_set, numsamps)

        osr_inds = [ind for ind, reg in enumerate(self.conn.region_labels) if reg in osr_list]
        return osr_list, osr_inds

    def select_ez_inside(self, numsamps):
        inside_list = np.random.choice(self.clinezregions, 
                size=min(len(self.clinezregions),numsamps), 
                replace=False)
        inside_inds = [ind for ind, reg in enumerate(self.conn.region_labels) if reg in inside_list]
        return inside_list, inside_inds
    
