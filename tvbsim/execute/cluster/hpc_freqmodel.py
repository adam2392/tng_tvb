from .hpc_basemodel import BaseHPC 
import numpy as np
import os

# Frequency models
from tvbsim.base.preprocess.dsp.frequencyanalysis import MorletWavelet
from tvbsim.base.preprocess.dsp.frequencyanalysis import MultiTaperFFT

''' Class wrappers for writing HPC mvar model computations '''
"""
Example usage:

hpcrunner = FFTModel(patient, winsize, stepsize, samplerate)
hpcrunner.loaddata(rawdata)
hpcrunner.loadmetadata(metadata)
hpcrunner.run(outputfilename, metafilename)

"""
class FreqModel(BaseHPC):
    rawdata = None
    winsize = None
    stepsize = None
    numwins = None     
    tempdir = NotImplementedError("Please set tempdir!")
    
    def __init__(self, patient, winsize, stepsize, samplerate, config=None):
        super(FreqModel, self).__init__(self, config)
        self.patient = patient          # patient identifier to analyze
        self.winsize = winsize          # winsize of model
        self.stepsize = stepsize        # stepsize of model
        self.samplerate = samplerate    # samplerate of data
    
    def get_tempfilename(tempfilename): 
        return os.path.join(self.tempdir, tempfilename)

    def loadmetafile(self, metafilename):
        self.metadata = self._loadjsonfile(metafilename)

    def loaddata(self, rawdata):
        # load raw data and other relevant channel data
        self.rawdata = rawdata
        self._computenumwins()

    def _computenumwins(self):
        # get number of channels and samples in the raw data
        numchans, numsignals = self.rawdata.shape
        # get number of samples in a window
        numwinsamps = self.winsize * int(self.samplerate) / 1000
        # get number of samples in a step
        numstepsamps = self.stepsize * int(self.samplerate) / 1000
        # get number of complete windows in raw data
        self.numwins = int(np.floor(numsignals / numstepsamps -
                                    numwinsamps / numstepsamps + 1))

class FFTModel(FreqModel):
    def __init__(self, patient, winsize, stepsize, samplerate, config=None):
        super(FFTModel, self).__init__(patient, winsize, stepsize, samplerate, config=config)
    
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

class MorletModel(FreqModel):
    def __init__(self, patient, winsize, stepsize, samplerate, config=None):
        super(MorletModel, self).__init__(patient, winsize, stepsize, samplerate, config=config)

         
    def run(self, outputfilename, metafilename):
        # Wavelet Parameters
        freqs = 2**(np.arange(1.,9.,1./5))
        waveletwidth = 6
        waveletargs = {"winsize": self.winsize,  
                        "stepsize": self.stepsize, 
                        "samplerate": self.samplerate, 
                        "waveletfreqs":waveletfreqs, 
                        "waveletwidth":waveletwidth}
        self.metadata['freqs'] = freqs
        self.metadata['waveletwidth'] = waveletwidth
        morlet = MorletWavelet(**waveletargs)

        ################################ 2. Run FFT Model ###########################
        morlet.loadrawdata(rawdata=rawdata)
        power, phase = morlet.multiphasevec()
        timepoints = morlet.timepoints
        samplepoints = morlet.samplepoints
        samplerate = morlet.samplerate

        # since morlet wavelet does not compress in time, we will
        # do it ourselves to save disk space
        power = _compress_windata(power,samplepoints,samplerate,winsize,stepsize)
        phase = _compress_windata(phase,samplepoints,samplerate,winsize,stepsize)

        self.metadata['timepoints'] = timepoints
        self.metadata['morletfilename'] = outputfilename
        # save the timepoints, included channels used, parameters
        self._writejsonfile(self.metadata, metafilename)
        self.logger.debug('Saved metadata as json!')

        # save adjacency matrix
              # save adjacency matrix
        self._writenpzfile(outputfilename, 
                            power=power,
                            freqs=freqs,
                            phase=phase)
        self.logger.debug("Saved morlet computation at {}".format(outputfilename))

