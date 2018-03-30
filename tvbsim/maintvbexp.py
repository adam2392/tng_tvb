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

    def initintegrator(self, ts=0.05, noise_cov=None, ntau=0, noiseon=True):
        if noise_cov is None:
            noise_cov = np.array([0.001, 0.001, 0.,
                                  0.0001, 0.0001, 0.])
        ####################### 3. Integrator for Models ##########################
        # define cov noise for the stochastic heun integrato
        hiss = noise.Additive(nsig=noise_cov)
        # hiss = noise.Additive(nsig=noise_cov, ntau=ntau)
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

    def setupsim(self, a=1., period=1., moved=False, initcond=None):
        ################## 4. Difference Coupling Between Nodes ###################
        coupl = coupling.Difference(a=a)
        # self.coupl = coupl

        if initcond is not None:
            self.initcond = initcond

        # either use gain file, or recompute it
        usegainfile = False
        if usegainfile:
            gainfile = self.gainfile
        else:
            gainfile = None

        # adding observation noise?
        # ntau=0
        # noise_cov=np.array([1.0])
        # obsnoise = noise.Additive(nsig=noise_cov, ntau=ntau)
        # obsnoise = None

        ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #############
        mon_tavg = monitors.TemporalAverage(period=period)  # monitor model

        if gainfile is None:
            mon_SEEG = monitors.iEEG.from_file(period=period,
                                               variables_of_interest=[1])
            # sensors_fname=self.seegfile,
            # rm_f_name=regmapfile,
            # projection_fname=gainfile)
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

        # initialize simulator object
        self.sim = simulator.Simulator(model=self.epileptors,
                                       # initial_conditions = self.init_cond,
                                       connectivity=self.conn,
                                       coupling=coupl,
                                       integrator=self.integrator,
                                       monitors=self.monitors)
        configs = self.sim.configure()
        return configs

    def mainsim(self, sim_length=60000):
        (times, epilepts), (_, seegts) = self.sim.run(
            simulation_length=sim_length)
        return times, epilepts, seegts
