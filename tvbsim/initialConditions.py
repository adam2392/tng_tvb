#!/usr/bin/env python2
import sys
import os.path
import numpy as np
from scipy.optimize import fsolve

# function to get model in its equilibrium value
def get_equilibrium(model, init):
	# the number of variables we need estimates for
    nvars = len(model.state_variables)
    cvars = len(model.cvar)


    def func(x):
    	x = x[0:nvars]
        fx = model.dfun(x.reshape((nvars, 1, 1)),
                        np.zeros((cvars, 1, 1)))
        return fx.flatten()

    x = fsolve(func, init)
    return x


    def dfun_test(self, x, c, local_coupling=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        Iext = self.Iext + local_coupling * x[0, :, 0]
        deriv = _numba_dfun(x_, c_,
                         self.x0, Iext, self.Iext2, self.a, self.b, self.slope, self.tt, self.Kvf,
                         self.c, self.d, self.r, self.Ks, self.Kf, self.aa, self.tau)
        return deriv.T[..., numpy.newaxis]
    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0,
             array=numpy.array, where=numpy.where, concat=numpy.concatenate):
        r"""
        Computes the derivatives of the state variables of the Epileptor
        with respect to time.

        Implementation note: we expect this version of the Epileptor to be used
        in a vectorized manner. Concretely, y has a shape of (6, n) where n is
        the number of nodes in the network. An consequence is that
        the original use of if/else is translated by calculated both the true
        and false forms and mixing them using a boolean mask.

        Variables of interest to be used by monitors: -y[0] + y[3]
        """
        y = state_variables
        ydot = numpy.empty_like(state_variables)

        Iext = self.Iext + local_coupling * y[0]
        c_pop1 = coupling[0, :]
        c_pop2 = coupling[1, :]

        # population 1
        if_ydot0 = - self.a*y[0]**2 + self.b*y[0]
        else_ydot0 = self.slope - y[3] + 0.6*(y[2]-4.0)**2
        ydot[0] = self.tt*(y[1] - y[2] + Iext + self.Kvf*c_pop1 + where(y[0] < 0., if_ydot0, else_ydot0) * y[0])
        ydot[1] = self.tt*(self.c - self.d*y[0]**2 - y[1])

        # energy
        if_ydot2 = - 0.1*y[2]**7
        else_ydot2 = 0
        ydot[2] = self.tt*(self.r * ( 4*(y[0] - self.x0) - y[2] + where(y[2] < 0., if_ydot2, else_ydot2) + self.Ks*c_pop1))

        # population 2
        ydot[3] = self.tt*(-y[4] + y[3] - y[3]**3 + self.Iext2 + 2*y[5] - 0.3*(y[2] - 3.5) + self.Kf*c_pop2)
        if_ydot4 = 0
        else_ydot4 = self.aa*(y[3] + 0.25)
        ydot[4] = self.tt*((-y[4] + where(y[3] < -0.25, if_ydot4, else_ydot4))/self.tau)

        # filter
        ydot[5] = self.tt*(-0.01*(y[5] - 0.1*y[0]))

        return ydot