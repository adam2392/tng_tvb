import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
from tvb.simulator.lab import *

def initintegrator(heun_ts, noise_cov, noiseon=True):
    ####################### 3. Integrator for Models ##########################
    # define cov noise for the stochastic heun integrato
    hiss = noise.Additive(nsig=noise_cov)

    if noiseon:
        heunint = integrators.HeunStochastic(dt=heun_ts, noise=hiss)
    else:
        heunint = integrators.HeunDeterministic(dt=heun_ts)

    return heunint