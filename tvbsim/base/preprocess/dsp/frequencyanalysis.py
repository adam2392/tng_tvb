import numpy as np
import scipy
import spectrum
from .helper import *
from .basefreqmodel import BaseFreqModel
import math
import fragility.base.constants.model_constants as constants

'''
ToDo: If you want to allow mtaper fft for any arbitrary off by factor of 0.5 in milliseconds
then we need to adapt the buffer function in fftchan.

Needs to slice through data correctly
'''


def next_greater_power_of_2(x):
    return 2**(x - 1).bit_length() if x != 0 else 1


class MultiTaperFFT(BaseFreqModel):
    def __init__(self, winsizems=constants.WINSIZE_SPEC, stepsizems=constants.STEPSIZE_SPEC,
                 samplerate=None, timewidth=constants.MTBANDWIDTH, method=None):
        BaseFreqModel.__init__(self, winsizems, stepsizems, samplerate)

        # multitaper FFT using welch's method
        self.timewidth = timewidth
        # possible values of method are 'eigen', 'hann',
        if not method:
            self.method = 'eigen'
            self.logger.info('Default method of tapering is eigen')
        self.freqsfft = np.linspace(
            0,
            self.samplerate // 2,
            (self.winsize * self.samplerate / 1000) // 2 + 1)

    def loadrawdata(self, rawdata):
        assert rawdata.shape[0] < rawdata.shape[1]
        # rem = rawdata.shape[1]%int(self.stepsamps)
        # if rem > 0:
        #     rawdata = rawdata[:,0:rawdata.shape[1]-rem]
        #     warnings.warn("Since stepsize is not an even cut through the data, \
        # we trimmed the data off at the end by " + str(rem) + " samples!")

        numsignals = rawdata.shape[1]
        self.compute_samplepoints(numsignals)
        self.compute_timepoints(numsignals)
        self.rawdata = rawdata
        self.logger.info("Loaded raw data in MultiTaperFFT!")

    def fftchan(self, datawin, numsamps=None, overlapsamps=None,
                numtapers=None, w=None, vweights=None):
        # split signal into windows and apply tapers to each one
        eegwin = self.buffer(datawin, numsamps, overlapsamps, opt='nodelay')
        detrendedeeg = scipy.signal.detrend(eegwin, axis=0)
        # need to adapt to repmat of matlab
        eegwin = np.repeat(
            detrendedeeg[:, :, np.newaxis], repeats=numtapers, axis=2)
        windows = eegwin.shape[1]
        wpermuted = np.transpose(
            np.repeat(w[:, :, np.newaxis], axis=2, repeats=windows), [0, 2, 1])

        # get coefficients, power and phases
        fx = np.fft.fft(np.multiply(wpermuted, eegwin), axis=0)

        # only get the frequencies we weant
        fx = fx[0:len(self.freqsfft), :, :] / np.sqrt(numsamps)

        # freq/window/taper to get the power
        fxpow = np.multiply(fx, np.conj(fx))
        fxpow = np.concatenate((fxpow[0, :, :][np.newaxis, :, :],
                                2 * fxpow[1:int(numsamps / 2), :, :],
                                fxpow[-1, :, :][np.newaxis, :, :]),
                               axis=0)
        fxphase = np.angle(fxpow)
        # assert 1==0
        # average over tapers, weighted by eigenvalues
        timefreqmat = np.mean(fxpow * vweights, axis=2)

        return timefreqmat, fxphase

    def mtwelch(self):
        # get dimensions of raw data
        numchans, numeegsamps = self.rawdata.shape

        # could BE A BUG FROM HARD CODING
        # get num samples for each FFT window and the freqs to get fft at
        numsamps = int(self.winsamps)
        overlapsamps = int(self.stepsamps)
        numwins = self.timepoints.shape[0]

        # set the number of tapers to use
        numtapers = 2 * self.timewidth - 1

        taperind = 1
        vweights = 1
        taperpownorm = 1
        taperampnorm = 1

        # get discrete tapering windows
        w, eigens = dpss_windows(numsamps, self.timewidth, numtapers)
        # get the weighted eigenvalues
        vweights = np.ones((1, 1, len(eigens)))
        vweights[0, 0, :] = eigens / np.sum(eigens)
        w = w.T  # transpose to make Freq X tapers

        powermultitaper = np.zeros(
            (numchans, len(self.freqsfft), numwins), dtype=complex)
        phasemultitaper = np.zeros((numchans, len(self.freqsfft), numwins))

        # loop through all the channels and compute the FFT
        for ichan in range(0, numchans):
            eegwin = self.rawdata[ichan, :]
            timefreqmat, fxphase = self.fftchan(eegwin, numsamps, overlapsamps,
                                                numtapers, w, vweights)

            if timefreqmat.shape[1] > numwins:
                if ichan == 0:
                    self.logger.info('Time freq mat getting messed up!')
                    self.logger.info(
                        'Time frequency matrix shape is: %s' % str(
                            timefreqmat.shape[1]))
                    self.logger.info(
                        'Numer of windows computed though is: %s' %
                        str(numwins))
                timefreqmat = timefreqmat[:, 0:-1]
                fxphase = fxphase[:, 0:-1, :]

            # average over windows and scale amplitude
            timefreqmat = timefreqmat * taperpownorm ** 2
            # save time freq data
            powermultitaper[ichan, :, :] = timefreqmat
            # save phase data - only of first taper -> can test complex average
            phasemultitaper[ichan, :, :] = fxphase[:, :, 0]

        # make it log based power
        powermultitaper = np.log10(powermultitaper)
        return powermultitaper, self.freqsfft, self.timepoints, phasemultitaper


class MorletWavelet(BaseFreqModel):
    def __init__(self, winsizems=constants.WINSIZE_SPEC, stepsizems=constants.STEPSIZE_SPEC,
                 samplerate=None, waveletfreqs=None, waveletwidth=constants.WAVELETWIDTH):
        BaseFreqModel.__init__(self, winsizems, stepsizems, samplerate)
        if waveletfreqs is None:
            self.logger.error('Wavelet freqs should be set here!')
        self.waveletfreqs = waveletfreqs
        self.waveletwidth = waveletwidth

    def loadrawdata(self, rawdata):
        assert rawdata.shape[0] < rawdata.shape[1]
        numchans = rawdata.shape[0]
        # buffer region of 1 second (milliseconds)
        self.bufferms = np.ceil(1000 * self.samplerate / 1000).astype(int)
        rawdata = np.concatenate((np.zeros((numchans, self.bufferms)),
                                  rawdata,
                                  np.zeros((numchans, self.bufferms))), axis=1)
        numsignals = rawdata.shape[1]
        self.compute_samplepoints(numsignals)
        self.compute_timepoints(numsignals)
        self.rawdata = rawdata
        self.logger.info("Loaded raw data in Morlet Transform!")

    def multiphasevec(self):
        # implement the multiphase vec
        nfreqs = len(self.waveletfreqs)
        nChans = self.rawdata.shape[0]
        nSamples = self.rawdata.shape[1]

        # initialze return arrays
        power = np.zeros((nChans, nfreqs, nSamples))
        phase = np.zeros((nChans, nfreqs, nSamples))

        # initialize step parameters
        dt = 1. / self.samplerate
        st = 1. / (2 * np.pi * self.waveletfreqs / self.waveletwidth)

        # use a list to store the waves at each frequency
        currwaves = []
        lencurrwaves = []
        # get the morlet's wavelet for each frequency

        def currwave_fun(i): return self.morlet(self.waveletfreqs[i],
                                                np.arange(-3.5 * st[i], 3.5 * st[i], dt))
        for i in range(0, nfreqs):
            # get the morlet wave, which will be uneven lengths
            currwave = currwave_fun(i)
            currwaves.append(currwave)
            lencurrwaves.append(len(currwave))
        lencurrwaves = np.array(lencurrwaves)

        # length of convolution of S and curwaves[i]
        lys = nSamples + lencurrwaves - 1
        lys2 = np.zeros((len(lys), 1))
        for i in range(0, len(lys)):
            lys2[i] = math.pow(2, spectrum.tools.nextpow2(lys[i]))

        # start index of signal after convolution and keep as int
        ind1 = np.ceil(lencurrwaves / 2).astype(int)

        # loop through and compute morlet transform for each frequency
        for idx, ly2 in enumerate(lys2):
            # convert to int, and get curwave
            ly2 = int(ly2.ravel())
            currwave = currwaves[idx]

            # Perform convolution of curwaves[i] with every row of S
            # then take FFT of S and curwaves[i], multiply, and iFFT
            sfft = np.fft.fft(a=self.rawdata, n=ly2, axis=1)
            currwavefft = np.fft.fft(a=currwave, n=ly2)
            Y = sfft * currwavefft
            y1 = np.fft.ifft(Y, ly2, axis=1)

            # get the correct y slices and get phase and power
            ''' CHECK INDICES? ARE THEY CORRECT FROM MATLAB -> PYTHON '''
            y1 = y1[:, ind1[idx] - 1: (ind1[idx] + nSamples - 1)]
            power[:, idx, :] = np.abs(y1**2)
            phase[:, idx, :] = np.angle(y1)
        # make it log based power
        power = np.log10(power)
        power = power[:, :, self.bufferms:-self.bufferms]
        phase = phase[:, :, self.bufferms:-self.bufferms]
        return power, phase

    def morlet(self, freq, time):
        '''
        # Morlet's wavelet for frequency f and time t.
        # The wavelet will be normalized so the total energy is 1.
        # width defines the ``width'' of the wavelet.
        # A value >= 5 is suggested.
        # Ref: Tallon-Baudry et al., J. Neurosci. 15, 722-734 (1997)
        @params:
        freq        (int)
        time        (int)

        # sf = f/width;
        # st = 1/(2*pi*sf);
        # A = 1/sqrt(st*sqrt(pi));
        # y = A*exp(-t.^2/(2*st^2)).*exp(i*2*pi*f.*t);
        '''
        sf = freq / self.waveletwidth
        st = 1 / (2 * np.pi * sf)
        A = 1 / np.sqrt(st * np.sqrt(np.pi))
        y = A * np.multiply(np.exp(-time ** 2 / (2 * st**2)),
                            np.exp(1j * 2 * np.pi * freq * time))
        return y
