import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
from tvb.simulator.lab import *
import numpy as np
from exp.basetvbexp import TVBExp
from exp.movecontactexp import MoveContactExp
import warnings

class MainTVBSim(TVBExp, MoveContactExp):
    def __init__(self, conn, condspeed=np.inf):
        TVBExp.__init__(self, conn=conn, condspeed=condspeed)

    def setezregion(self, ezregions, rand=False):
        if np.asarray(ezregions).size == 1:
            self.ezind = np.array(self._getindexofregion(ezregions))
            self.ezregion = np.array(ezregions)
        else:
            ezinds = []
            ezregs = []
            for ezreg in ezregions:
                ezregs.append(ezreg)
                ezinds.append(self._getindexofregion(ezreg))
            self.ezind = np.array(ezinds)
            self.ezregion = np.array(ezregs)
        if rand==True:
            self.ezind, self.ezregion = self.sample_randregions(1)
    def setpzregion(self, pzregions, rand=False):
        if np.asarray(pzregions).size == 1:
            self.pzind = np.array(self._getindexofregion(pzregion))
            self.pzregion = np.array(pzregions)
        elif np.asarray(pzregions).size > 1:
            pzinds = []
            pzregs = []
            for pzreg in pzregions:
                pzregs.append(pzreg)
                pzinds.append(self._getindexofregion(pzreg))
            self.pzind = np.array(pzinds)
            self.pzregion = np.array(pzregs)
        else:
            self.pzind = np.array([])
            self.pzregion = None
        if rand==True:
            self.pzind, self.pzregion = self.sample_randregions(1)

    def initintegrator(self, ts=0.05, noise_cov=None, ntau=0, noiseon=True):
        if noise_cov is None:
            noise_cov = np.array([0.001, 0.001, 0.,\
                                  0.0001, 0.0001, 0.])
        ####################### 3. Integrator for Models ##########################
        # define cov noise for the stochastic heun integrato
        hiss = noise.Additive(nsig=noise_cov, ntau=ntau)
        # hiss = noise.Multiplicative(nsig=noise_cov)
        if noiseon:
            heunint = integrators.HeunStochastic(dt=ts, noise=hiss)
        else:
            heunint = integrators.HeunDeterministic(dt=ts)
        self.integrator = heunint

    def initepileptor(self, x0norm, x0ez=None, x0pz=None, r=None, Ks=None, tt=None, tau=None):
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
        epileptors = models.Epileptor(
                        variables_of_interest=['z', 'x2-x1'])
        if r is not None:
            epileptors.r = r
        if Ks is not None:
            epileptors.Ks = Ks
        if tt is not None:
            epileptors.tt = tt
        if tau is not None:
            epileptors.tau = tau

        # this comes after setting all parameters
        epileptors.x0 = x0norm*np.ones(len(self.conn.region_labels))
        if x0ez is not None:
            try:
                epileptors.x0[self.ezind] = x0ez
            except AttributeError:
                sys.stderr.write("EZ index not set yet! Do you want to proceed with simulation?")
                warnings.warn("EZ index not set yet! Do you want to proceed with simulation?")
        if x0pz is not None:
            try:
                epileptors.x0[self.pzind] = x0pz
            except AttributeError:
                sys.stderr.write("PZ index not set yet! Do you want to proceed with simulation?")
                warnings.warn("pz index not set yet! Do you want to proceed with simulation?")
        self.epileptors = epileptors

    def setupsim(self,a=1.,period=1.,moved=False,regmapfile=None):
        ################## 4. Difference Coupling Between Nodes ###################
        coupl = coupling.Difference(a=a)
        # self.coupl = coupl
        ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #############
        mon_tavg = monitors.TemporalAverage(period=period) # monitor model
        mon_SEEG = monitors.iEEG.from_file(period=period,
                                           variables_of_interest=[1])
                                           # sensors_fname=self.seegfile,
                                           # rm_f_name=regmapfile,
                                           # projection_fname=self.gainfile)
        sim_monitors = [mon_tavg, mon_SEEG]
        # set to the object's seeg xyz and gain mat
        # if moved:
        sim_monitors[1].sensors.locations = self.seeg_xyz
        sim_monitors[1].gain = self.gainmat
        self.monitors = sim_monitors

        # initialize simulator object
        self.sim = simulator.Simulator(model = self.epileptors,
                                  initial_conditions = self.init_cond,
                                  connectivity = self.conn,
                                  coupling = coupl,
                                  integrator = self.integrator,
                                  monitors = self.monitors)
        configs = self.sim.configure()
        return configs

    def mainsim(self, sim_length=60000):
        (times, epilepts), (_, seegts) = self.sim.run(simulation_length=sim_length)
        return times, epilepts, seegts
