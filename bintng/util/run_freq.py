import os
import json
import numpy as np
import io
from fragility.base.utils.data_structures_utils import NumpyEncoder
from fragility.base.preprocess.dsp.frequencyanalysis import MorletWavelet
from fragility.base.preprocess.dsp.frequencyanalysis import MultiTaperFFT
try:
    to_unicode = unicode
except NameError:
    to_unicode = str
    
def _compress_windata(data, samplepoints, samplerate, winsize, stepsize):
    '''
    Compresses the data with a window/stepsize specified by user
    over the sample points
    '''
    numchans, numfreqs, _ = data.shape
    numwins = samplepoints.shape[0]
    compressed_data = np.zeros((numchans, numfreqs, numwins))

    # loop through each window and compress the data with average
    for iwin in range(numwins):
        compressed_data[:,:,iwin] = np.mean(data[:,:,samplepoints[iwin,0]:samplepoints[iwin,1]], axis=-1)
    return compressed_data

class FreqAnalysis(object):
    @staticmethod
    def run_morlet(rawdata, waveletargs):
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
        
        print(power.shape)
        print(phase.shape)

        return power, phase, timepoints

    @staticmethod
    def run_fft(rawdata, fftargs):
        mtaper = MultiTaperFFT(**fftargs)

        ################################ 2. Run FFT Model ###########################
        mtaper.loadrawdata(rawdata=rawdata)
        power, freqs, timepoints, phase = mtaper.mtwelch()

        return power, freqs, phase, timepoints

    @staticmethod
    def save_data(outputfilename, outputmetafilename, power, phase, metadata):
        # save the output computation into one file
        np.savez_compressed(outputfilename, 
                            power=power, 
                            phase=phase)
        # save the timepoints, included channels used, parameters
        try:
            # save the timepoints, included channels used, parameters
            dumped = json.dumps(metadata, cls=NumpyEncoder)
            with open(outputmetafilename, 'w') as f:
                json.dump(dumped, f)
        except Exception as e:
            dumped = json.dumps(metadata, cls=NumpyEncoder)
            with io.open(outputmetafilename, 'w', encoding="utf-8") as f:
                json.dump(dumped, f)
        print('Saved metadata as json!')
