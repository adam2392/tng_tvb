import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
from tvb.simulator.lab import *

def initcoupling(a):
    ################## 4. Difference Coupling Between Nodes ###################
    # define a simple difference coupling
    coupl = coupling.Difference(a=1.)
    return coupl