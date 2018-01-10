import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
from tvb.simulator.lab import *

def initmonitors(period, seegfile, gainmatfile, varindex):
    ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #############
    mon_tavg = monitors.TemporalAverage(period=period) # monitor model
    mon_SEEG = monitors.iEEG.from_file(sensors_fname=seegfile,
                        projection_fname=gainmatfile,
                        period=period,
                        variables_of_interest=[1]
                    )

    return [mon_tavg, mon_SEEG]