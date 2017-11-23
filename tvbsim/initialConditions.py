#!/usr/bin/env python2
import sys
import os.path
import numpy as np
from scipy.optimize import fsolve

# function to get model in its equilibrium value
def get_equilibrium(model, init):
    nvars = len(model.state_variables)
    cvars = len(model.cvar)

    def func(x):
        fx = model.dfun(x.reshape((nvars, 1, 1)),
                        np.zeros((cvars, 1, 1)))
        return fx.flatten()

    x = fsolve(func, init)
    return x