import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
from tvb.simulator.lab import *
import numpy as np
from exp.basetvbexp import TVBExp
from exp.movecontactexp import MoveContactExp
import warnings

class MainTVBSim(TVBExp, MoveContactExp):
    x0ez = x0pz = x0norm = None
    ezindices = []
    
    def __init__(self, conn, condspeed=np.inf):
        TVBExp.__init__(self, conn=conn, condspeed=condspeed)

    def get_metadata(self):
        self.metadata = {
                'regions': self.conn.region_labels,
                'regions_centers': self.conn.centres,
                'chanlabels': self.seeg_labels,
                'chanxyz': self.seeg_xyz,
                'ezregs': self.ezregion,
                'pzregs': self.pzregion,
                'ezindices': self.ezind,
                'pzindices': self.pzind,
                'epiparams': self.getepileptorparams(),
                'gainmat': self.gainmat,
                'x0ez': self.x0ez,
                'x0pz': self.x0pz,
                'x0norm': self.x0norm,
        }
        return self.metadata

    def setezregion(self, ezregions, rand=False):
        if np.asarray(ezregions).size == 1:
            ezregions = np.array(ezregions)[0]
            ezind = np.array([self._getindexofregion(ezregions)])
            ezregion = np.array(ezregions)

        elif np.asarray(ezregions).size > 1:
            ezinds = []
            ezregs = []
            for ezreg in ezregions:
                ezregs.append(ezreg)
                ezinds.append(self._getindexofregion(ezreg))

            ezind = np.array(ezinds)
            ezregion = np.array(ezregs)
        else:
            ezind = []
            ezregion = None

        # if np.asarray(ezind).size == 1:
        #     ezind = [ezind]
        self.ezind = ezind
        self.ezregion = ezregion
        if rand == True:
            self.ezind, self.ezregion = self.sample_randregions(1)

    def setpzregion(self, pzregions, rand=False):
        if np.asarray(pzregions).size == 1:
            pzregions = np.array(pzregions)[0]
            pzind = np.array([self._getindexofregion(pzregions)])
            pzregion = np.array(pzregions)

        elif np.asarray(pzregions).size > 1:
            pzinds = []
            pzregs = []
            for pzreg in pzregions:
                pzregs.append(pzreg)
                pzinds.append(self._getindexofregion(pzreg))

            pzind = np.array(pzinds)
            pzregion = np.array(pzregs)
        else:
            pzind = []
            pzregion = None

        # if np.asarray(pzind).size == 1:
        #     pzind = [pzind]
        self.pzind = pzind
        self.pzregion = pzregion

        if rand == True:
            self.pzind, self.pzregion = self.sample_randregions(1)

    def loadintegrator(self, integrator_params):
        # heunint = integrators.HeunStochastic(**integrator_params)
        heunint = integrators.HeunDeterministic(**integrator_params)
        self.integrator = heunint

    def loadepileptor(self, ezregions, pzregions,
                        x0ez=-2.3, x0pz=-2.05, x0norm=-1.6, epileptor_params=None):
        '''
        State variables for the Epileptor model:
        Repeated here for redundancy:
        x1 = first
        y1 = second
        z = third
        x2 = fourth
        y2 = fifth
        '''
        if epileptor_params is None:
            epileptor_params = {
                    'r': 0.00037,#/1.5   # Temporal scaling in the third state variable
                    'Ks': -10,                 # Permittivity coupling, fast to slow time scale
                    'tt': 0.07,                   # time scale of simulation
                    'tau': 10,                   # Temporal scaling coefficient in fifth st var
                    'x0': -2.45, # x0c value = -2.05
                    # 'Iext': iext,
                }
            print("In maintvbexp.py using default parameters!")
        
        self.setezregion(ezregions)
        self.setpzregion(pzregions)
        ####################### 2. Neural Mass Model @ Nodes ##################
        epileptors = models.Epileptor(variables_of_interest=['z', 'x2-x1', 'x1', 'x2', 'y1', 'y2', 'g'], **epileptor_params)

        # this comes after setting all parameters
        epileptors.x0 = x0norm * np.ones(len(self.conn.region_labels))
        if x0ez is not None:
            try:
                epileptors.x0[self.ezind] = x0ez
            except AttributeError:
                sys.stderr.write(
                    "EZ index not set yet! Do you want to proceed with simulation?")
                warnings.warn(
                    "EZ index not set yet! Do you want to proceed with simulation?")
        if x0pz is not None:
            try:
                epileptors.x0[self.pzind] = x0pz
            except AttributeError:
                sys.stderr.write(
                    "PZ index not set yet! Do you want to proceed with simulation?")
                warnings.warn(
                    "pz index not set yet! Do you want to proceed with simulation?")

        self.epileptors = epileptors

    def loadcoupling(self, a=1.):
        ################## 4. Difference Coupling Between Nodes ###############
        coupl = coupling.Difference(a=a)
        self.coupl = coupl

    def loadmonitors(self, period=1., moved=False, initcond=None):
        if initcond is not None:
            self.initcond = initcond

        # either use gain file, or recompute it
        usegainfile = True
        if usegainfile:
            gainfile = self.gainfile
        else:
            gainfile = None

        ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #########
        mon_tavg = monitors.TemporalAverage(period=period)  # monitor model

        if gainfile is None:
            mon_SEEG = monitors.iEEG.from_file(period=period,
                                               variables_of_interest=[1])
        else:
            mon_SEEG = monitors.iEEG.from_file(period=period,
                                               variables_of_interest=[1],
                                               # sensors_fname=self.seegfile,
                                               # rm_f_name=regmapfile,
                                               projection_fname=gainfile)
        sim_monitors = [mon_tavg, mon_SEEG]
        # set to the object's seeg xyz and gain mat
        # if moved:
        sim_monitors[1].sensors.locations = self.seeg_xyz

        if gainfile is None:
            self.gainmat = self.gain_matrix_inv_square()
            # self.gainmat = self.simplest_gain_matrix()
            sim_monitors[1].gain = self.gainmat
        self.monitors = sim_monitors

    def setupsim(self):
        # initialize simulator object
        simulator_params = {
            'model': self.epileptors,
            'connectivity': self.conn,
            'coupling': self.coupl,
            'integrator': self.integrator,
            'monitors': self.monitors
        }
        self.sim = simulator.Simulator(**simulator_params)
        configs = self.sim.configure()
        return configs

    def mainsim(self, sim_length=60000):
        (times, simvars), (_, seegts) = self.sim.run(simulation_length=sim_length)
        return times, simvars, seegts
