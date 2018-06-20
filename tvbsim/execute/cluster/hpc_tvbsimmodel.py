from .hpc_basemodel import BaseHPC 
import numpy as np
import os

# Wrappers for tvb model
from tvb.simulator.lab import *

from tvbsim.io.simdata.loadtvbdata import StructuralDataLoader
from tvbsim.io.patient.subject import Subject
from tvbsim.exp.utils.selectregion import Regions
from tvbsim.maintvbexp import MainTVBSim

from tvbsim.postprocess.postprocess import PostProcessor
from tvbsim.postprocess.detectonsetoffset import DetectShift
''' Class wrappers for writing HPC mvar model computations '''
"""
Example usage:

hpcrunner = SimvsRealModel(patient, period, sim_length)
hpcrunner.loadalldata(root_dir, ezinside=False)
hpcrunner.initialize_tvb_model()
hpcrunner.run(outputfilename, metafilename)


"""
class TVBSimModel(BaseHPC):
    rawdata = None
    period = None
    samplerate = None 
    sim_length = None  
    tempdir = NotImplementedError("Please set tempdir!")
    
    def __init__(self, patient, period, sim_length, config=None):
        super(TVBSimModel, self).__init__(config)
        self.patient = patient          # patient identifier to analyze
        self.period = period
        self.sim_length = sim_length    # samplerate of data
        
    def loadmetafile(self, metafilename):
        self.metadata = self._loadjsonfile(metafilename)

    def loadalldata(self, root_dir, ezinside, 
            shufflepats=False, shuffleweights=False, numsamps=2):
        self.root_dir = root_dir

        loader = Subject(name=self.patient, root_dir=os.path.join(root_dir, self.patient), preload=False)
        self.conn = connectivity.Connectivity.from_file(loader.connfile)
        self.surf = loader.surf 
        self.gainfile = loader.gainfile
        self.sensorsfile = loader.sensorsfile
        self.chanxyzlabels = loader.chanxyzlabels

        # get the ez/pz indices we want to use
        self.clinezinds = loader.ezinds
        self.clinpzinds = []
        self.clinezregions = list(loader.conn.region_labels[self.clinezinds])
        self.clinpzregions = []

        if ezinside:
            ezregs, ezinds = self.select_ez_inside(numsamps=numsamps)
        else:
            ezregs, ezinds = self.select_ez_outside(numsamps=numsamps)

        self.modelezinds = ezinds
        self.modelpzinds = []
        self.modelezregions = ezregs
        self.modelpzregions = []

        print("Model ez: ", self.modelezregions, self.modelezinds)
        print("Model pz: ", self.modelpzregions, self.modelpzinds)
       
        if shuffleweights and shufflepats:
            raise AttributeError("Shuffle weights and patients cant both be set.")
        if shuffleweights and not shufflepats:
            self.conn, randpat = self.shuffleweights()
        elif shufflepats and not shuffleweights:
            self.conn, randpat = self.shufflepatients()

    def shuffleweights(self):
        # shuffle within patients
        randweights = MainTVBSim.randshuffleweights(self.conn.weights)
        self.conn.weights = randweights
        randpat = None
        return conn, randpat

    def shufflepatients(self):   
        # shuffle across patients
        randpat = MainTVBSim.randshufflepats(allpats, patient)   
        shuffled_connfile = os.path.join(self.root_dir, randpat, 'tvb', 'connectivity.zip')
        if not os.path.exists(shuffled_connfile):
            shuffled_connfile = os.path.join(self.root_dir, randpat, 'tvb', 'connectivity.dk.zip')
        conn = connectivity.Connectivity.from_file(shuffled_connfile)
        return conn, randpat
             
    def select_ez_outside(self,conn, clinezregions, numsamps):
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
    
class SimVsRealModel(TVBSimModel):
    samplerate = 1000

    def __init__(self, patient, period, sim_length, config=None):
        super(SimVsRealModel, self).__init__(patient, period, sim_length, config=config)

    def move_contacts(self, movedist):
        # move contacts if we wnat to
        for ind in self.maintvbexp.ezind:
            new_seeg_xyz, elecindicesmoved = self.maintvbexp.move_electrodetoreg(ind, movedist)
            print(elecindicesmoved)
            print(maintvbexp.seeg_labels[elecindicesmoved])
        print("Moved electrodes!")

    def run(self, outputfilename, metafilename):
        ######################## run simulation ########################
        configs = self.maintvbexp.setupsim()
        times, statevars_ts, seegts = self.maintvbexp.mainsim(sim_length=int(self.sim_length))

        # postprocessing
        epits, seegts, zts, state_vars = self._postprocess_sim(times, statevars_ts, seegts)
        metadata = self._postprocess_metadata()
        metadata['simfilename'] = outputfilename
        # save metadata
        self._writejsonfile(metadata, metafilename)

        # save simulation results
        np.savez_compressed(outputfilename, epits=epits, 
                                seegts=seegts,
                                times=times, 
                                zts=zts, 
                                state_vars=state_vars)
        return epits, seegts, zts, state_vars

    def initialize_tvb_model(self, **kwargs):
        self.maintvbexp = MainTVBSim(self.conn, condspeed=np.inf)
        # load the necessary data files to run simulation
        self.maintvbexp.loadseegxyz(seegfile=self.sensorsfile)
        self.maintvbexp.loadgainmat(gainfile=self.gainfile)
        self.maintvbexp.importsurfdata(surf=self.surf)
        ######### Model (Epileptor) Parameters ##########
        epileptor_params = {
            'r': 0.00037,#/1.5   # Temporal scaling in the third state variable
            'Ks': -10,                 # Permittivity coupling, fast to slow time scale
            'tt': 0.07,                   # time scale of simulation
            'tau': 10,                   # Temporal scaling coefficient in fifth st var
            'x0': -2.45, # x0c value = -2.05
        }

        for key, value in kwargs.iteritems():
            print "%s = %s" % (key, value)
            if key == 'Iext' or key == 'eps1':
                epileptor_params[key] = value

        ######### Integrator Parameters ##########
        ntau = 0
        noise_cov = np.array([0.001, 0.001, 0.,\
                                  0.0001, 0.0001, 0.])
        # define cov noise for the stochastic heun integrator
        hiss = noise.Additive(nsig=noise_cov, ntau=ntau)
        # hiss = noise.Multiplicative(nsig=noise_cov)
        integrator_params = {
            'dt': 0.05,
            'noise': hiss,
        }

        # load couping
        coupling_params = {
            'a': 1.,
        }
        # load monitors
        initcond = None
        monitor_params = {
            'period': self.period,
            'moved': False,
            'initcond': initcond
        }
        # initialize the entire pipeline
        self._init_epileptor(**epileptor_params)
        self._init_integrator(**integrator_params)
        self._init_coupling(**coupling_params)
        self._init_monitors(**monitor_params)

        self.allindices = np.hstack((self.maintvbexp.ezind, 
                            self.maintvbexp.pzind)).astype(int) 
        
    def _init_epileptor(self, **epileptor_params):
        x0ez=-1.65
        x0pz=-2.0 # x0pz = None
        if self.modelezregions is None:
            x0ez = None
        if self.modelpzregions is None:
            x0pz = None
        self.maintvbexp.loadepileptor(ezregions=self.modelezregions, 
                                pzregions=self.modelpzregions,
                                x0ez=x0ez, x0pz=x0pz,
                                epileptor_params=epileptor_params)
    def _init_integrator(self, **integrator_params):
        self.maintvbexp.loadintegrator(**integrator_params)

    def _init_coupling(self, **coupling_params):
        self.maintvbexp.loadcoupling(**coupling_params)

    def _init_monitors(self, **monitor_params):
        self.maintvbexp.loadmonitors(**monitor_params)
    
    def _postprocess_sim(self, times, statevars_ts, seegts):
        ######################## POST PROCESSING ########################
        postprocessor = PostProcessor(samplerate=self.samplerate, allszindices=self.allindices)
        secstoreject = 15
        times, epits, seegts, zts, state_vars = postprocessor.postprocts(statevars_ts, seegts, times, secstoreject=secstoreject)

        # GET ONSET/OFFSET OF SEIZURE
        detector = DetectShift()
        settimes = detector.getonsetsoffsets(epits, self.allindices)
        self.seizonsets, self.seizoffsets = detector.getseiztimes(settimes)
        print("The detected onset/offsets are: {}".format(zip(self.seizonsets,self.seizoffsets)))
        return epits, seegts, zts, state_vars

    def _postprocess_metadata(self):
        # save metadata from the exp object and from here
        metadata = self.maintvbexp.get_metadata()
        metadata['patient'] = self.patient
        metadata['samplerate'] = self.samplerate
        metadata['clinez'] = self.clinezregions
        metadata['clinpz'] = self.clinpzregions
        metadata['onsettimes'] = self.seizonsets
        metadata['offsettimes'] = self.seizoffsets

        return metadata
