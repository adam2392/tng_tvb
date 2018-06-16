from .hpc_basemodel import BaseHPC 
import numpy as np
import os

# Wrappers for tvb model
from tvb.simulator.lab import *
from tvbsim.maintvbexp import MainTVBSim
''' Class wrappers for writing HPC mvar model computations '''
"""
Example usage:

hpcrunner = FFTModel(patient, winsize, stepsize, samplerate)
hpcrunner.loaddata(rawdata)
hpcrunner.loadmetafile(metadata)
hpcrunner.run(outputfilename, metafilename)

"""
class TVBSimModel(BaseHPC):
    rawdata = None
    period = None
    samplerate = None   
    tempdir = NotImplementedError("Please set tempdir!")
    
    def __init__(self, patient, period, samplerate, config=None):
        super(TVBSimModel, self).__init__(self, config)
        self.patient = patient          # patient identifier to analyze
        self.period = period
        self.samplerate = samplerate    # samplerate of data
    
    def loadmetafile(self, metafilename):
        self.metadata = self._loadjsonfile(metafilename)

    def load_conn(self, connfile):
        # load raw data and other relevant channel data
        self.conn = pass

    def load_surf(self, surfacefile):
        pass

    def load_sensors(self, sensorsfile):
        pass

    def load_gain(self, gainfile):
        pass

        
class SimVsRealModel(TVBSimModel):
    def __init__(self, patient, period, samplerate, config=None):
        super(SimVsRealModel, self).__init__(patient, period, samplerate, config=config)
    
    def save_processed_data(filename, state_vars, seegts):
        print('finished simulating!')
        print(epits.shape)
        print(seegts.shape)
        print(times.shape)
        print(zts.shape)
        print(state_vars.keys())

        # save tseries
        np.savez_compressed(filename, epits=epits, 
                                    seegts=seegts,
                                    times=times, 
                                    zts=zts, 
                                    state_vars=state_vars)


    def run(self, outputfilename, metafilename):
        ############## RUN MODEL(S) ###############
        # instantiate a mvarmodel object
        mvarargs = { 
            "winsize": self.winsize,  
            "stepsize": self.stepsize, 
            "samplerate": self.samplerate, 
        }
        # FFT Parameters
        mtbandwidth = 4
        mtfreqs = []
        fftargs = {"winsize": self.winsize,  
                    "stepsize": self.stepsize, 
                    "samplerate": self.samplerate,  
                    "mtfreqs":mtfreqs, 
                    "mtbandwidth":mtbandwidth}
        # add to metadata
        self.metadata['mtbandwidth'] = mtbandwidth
        self.metadata['freqs'] = freqs

        mtaper = MultiTaperFFT(**fftargs)
        ################################ 2. Run FFT Model ###########################
        mtaper.loadrawdata(rawdata=rawdata)
        power, freqs, timepoints, phase = mtaper.mtwelch()

        self.metadata['timepoints'] = timepoints
        self.metadata['fftfilename'] = outputfilename
        # save the timepoints, included channels used, parameters
        self._writejsonfile(self.metadata, metafilename)
        self.logger.debug('Saved metadata as json!')

        # save adjacency matrix
        self._writenpzfile(outputfilename, 
                            power=power,
                            freqs=freqs,
                            phase=phase)
        self.logger.debug("Saved fft computation at {}".format(outputfilename))
