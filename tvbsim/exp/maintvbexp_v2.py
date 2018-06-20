import numpy as np
import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
from tvb.simulator.lab import *
from tvbsim.exp.base import BaseTVBExp
import warnings

class MainTVBSim(BaseTVBExp):
    x0ez = x0pz = x0norm = None
    ezindices = []

    coupl = None
    epileptors = None
    conn = None
    monitors = None
    integrators = None
    
    def __init__(self, config=None):
        super(TVBExp, self).__init__(config=config)

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

    def loadconn(self, conn, condspeed):
        self.conn = conn
        self.conn.speed = condspeed
        self.conn.cortical[:] = True
        self.conn.weights = conn.weights / np.max(conn.weights)

    def loadintegrator(self, dt, noise):
        heunint = integrators.HeunStochastic(dt=dt, noise=noise)
        # heunint = integrators.HeunDeterministic(**integrator_params)
        self.integrator = heunint

    def loadmodel(self, ezregions, pzregions, x0ez, x0pz, x0norm, 
        r, Ks, tt, tau, Iext, eps1):
        epileptor_params = {
                    'r': r,       # Temporal scaling in the third state variable
                    'Ks': Ks,                 # Permittivity coupling, fast to slow time scale
                    'tt': tt,                   # time scale of simulation
                    'tau': tau,                   # Temporal scaling coefficient in fifth st var
                    'x0': x0norm, # x0c value = -2.05
                    'Iext': Iext,
                    'eps1': eps1,
                }
        self.setezregion(ezregions)
        self.setpzregion(pzregions)
        ####################### 2. Neural Mass Model @ Nodes ##################
        epileptors = models.Epileptor(variables_of_interest=['z', 'x2-x1', 'x1', 
                                                            'x2', 'y1', 'y2', 'g'], 
                                    **epileptor_params)
        # this comes after setting all parameters
        epileptors.x0 = x0norm * np.ones(len(self.conn.region_labels))
        if x0ez is not None and ezregions is not None:
            epileptors.x0[self.ezind] = x0ez
        if x0pz is not None and pzregions is not None:
            epileptors.x0[self.pzind] = x0pz
        self.epileptors = epileptors

    def loadcoupling(self, type_cpl='diff', a=1.):
        ################## 4. Difference Coupling Between Nodes ###############
        if type_cpl == 'diff':
            coupl = coupling.Difference(a=a)
        elif type_cpl == 'linear':
            coupl = coupling.Linear(a=a, b=b)
        elif type_cpl == 'sigmoidal':
            coupl = coupling.Sigmoidal(a=a)
        elif type_cpl == 'hyperbolictangent':
            coupl = coupling.HyperbolicTangent(a=a)
        else:
            self.logger.error("Coupling type can only be diff, linear, sigmoidal, or HyperbolicTangent!")
        self.coupl = coupl

    def loadmonitors(self, chanxyz, gainmat, period=1., moved=False, initcond=None):
        if initcond is not None:
            self.initcond = initcond

        ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #########
        mon_tavg = monitors.TemporalAverage(period=period)  # monitor model

        mon_SEEG = monitors.iEEG.from_file(period=period,
                                variables_of_interest=[1])
                               # sensors_fname=self.seegfile,
                               # rm_f_name=regmapfile,
                               # projection_fname=gainfile)
        sim_monitors = [mon_tavg, mon_SEEG]
        # set to the object's seeg xyz and gain mat
        sim_monitors[1].sensors.locations = chanxyz
        sim_monitors[1].gain = gainmat

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
