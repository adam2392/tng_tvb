from tvbsim.base.preprocess.mne.base import BaseFreqModel
import numpy as np
import mne

"""
    TODO: Compute samplepoints for a given winsize/stepsize
    - morlet
    - multitaper morlet
    - for the stft as well

Possibly just drop last and first windows, since those are the ones that are 
zero padded. 

Resolution problem at millisecond scale?
    
"""

class FreqAnalysis(BaseFreqModel):
    fmin = None
    fmax = None
    n_cycles = None
    mtbandwidth = 4

    def __init__(self, winsize, stepsize, samplerate, freqs=None, config=None):
        super(FreqAnalysis, self).__init__(winsize, stepsize, samplerate, config)

        self.freqs = freqs 
        self.n_cycles = 7

    def run(self, rawdata):
        ################################ 2. Run FFT Model ###########################
        power, freqs = self.tfr(rawdata, psdtype='stft')

        return power, freqs

    def _findtimewins(self, times):
        indices = []
        for time in ensure_list(times):
            if time == 0:
                indices.append(time)
            else:
                idx = (time >= self.timepoints[:,0])*(time <= self.timepoints[:,1])
                timeind = np.where(idx)[0]
                if len(timeind) > 0:
                    indices.append(timeind[0])
                else:
                    indices.append(np.nan)
        return indices

    def mapinds_towins(self, inds):
        self.timepoints = np.array(self.timepoints)

        if inds is None:
            return None
        # get the actual indices that occur within time windows
        winind = self._findtimewins(inds)
        return winind

    def binfreqvalues(self, power, freqs, freqbands):
        if power.ndim == 2:
            power = power[...,np.newaxis]

        power = np.abs(power)
        # Create an empty array
        power_binned = np.zeros(shape=(power.shape[0],
                                       len(freqbands),
                                       power.shape[2]))
        
        for idx, (name, freqband) in enumerate(sorted(freqbands.items())):
            print(name, freqband)
            # compute the freq indices for each band
            freqbandindices = self._computefreqindices(freqs, freqband)
            
            # Create an empty array = C x T (frequency axis is compresssed into 1 band)
            # average between these two indices
            power_binned[:, idx, :] = np.mean(
                power[:, freqbandindices[0]:freqbandindices[1] + 1, :], axis=1)
        return power_binned

    def _computefreqindices(self, freqs, freqband):
        """
        Compute the frequency indices for this frequency band

        freqs = list of frequencies
        freqband = [lowerbound, upperbound] frequencies of the 
                frequency band
        """
        for freq in freqband:
            lowerband = freqband[0]
            upperband = freqband[1]

            # get indices where the freq bands are put in
            freqbandindices = np.where(
                (freqs >= lowerband) & (freqs < upperband))
            freqbandindices = [freqbandindices[0][0], freqbandindices[0][-1]]
        return freqbandindices

    def compute_timepoints_fromsamps(self):
        timepoints = np.divide(self.samplepoints, self.samplerate)
        self.timepoints = np.multiply(timepoints, 1000)
        
    def compute_samplepoints(self, numtimepoints):
        n_step = int(np.ceil(numtimepoints / float(self.stepsize)))
        # Zero-padding and Pre-processing for edges
        numtimepoints = self.winsize + (n_step - 1) * self.stepsize

        # Creates a [n,2] array that holds the sample range of each window that
        # is used to index the raw data for a sliding window analysis
        samplestarts = np.arange(0, numtimepoints - self.winsize + 1., 
                                    self.stepsize).astype(int)
        sampleends = np.arange(self.winsize - 1., numtimepoints,
                                    self.stepsize).astype(int)
        samplepoints = np.append(samplestarts[:, np.newaxis],
                                 sampleends[:, np.newaxis], axis=1)
        self.samplepoints = samplepoints

        self.compute_timepoints_fromsamps()
        return samplepoints

    def _setmtaper(self, mtbandwidth, freqs):
        self.mtbandwidth = mtbandwidth
        self.freqs = freqs
        self.n_cycles = freqs / 2

    def _setwavelet(self, waveletfreqs, waveletwidth):
        # Wavelet Parameters
        waveletfreqs = 2**(np.arange(1.,9.,1./5))
        waveletwidth = 6 # None
        wavelets = mne.time_frequency.morlet(sfreq=self.samplerate, 
                                freqs=waveletfreqs, sigma=waveletwidth)
        return wavelets

    def psd(self, data, fmin, fmax, psdtype='mtaper'):
        if psdtype == 'mtaper':
            power, freqs = mne.time_frequency.psd_array_multitaper(data, 
                                                    sfreq=self.samplerate, 
                                                    fmin=fmin, 
                                                    fmax=fmax,
                                                    bandwidth=self.mtbandwidth)
        elif psdtype == 'welch':
            power, freqs = mne.time_frequency.psd_array_welch(data, 
                                                    sfreq=self.samplerate, 
                                                    fmin=fmin, 
                                                    fmax=fmax)

        power = np.log10((power))
        return power, freqs

    def tfr(self, data, psdtype='mtaper'):
        if data.ndim == 2:
            data = data[np.newaxis,...]

        if psdtype == 'mtaper':
            freqs = self.freqs
            power = mne.time_frequency.tfr_array_multitaper(data, 
                                                       sfreq=self.samplerate, 
                                                       freqs=self.freqs, 
                                                       n_cycles=self.n_cycles,
                                                       bandwidth=self.mtbandwidth)
        elif psdtype == 'stft':
            data = data.squeeze()
            power = mne.time_frequency.stft(data, wsize=self.winsize, tstep=self.stepsize)
            freqs = mne.time_frequency.stftfreq(wsize=self.winsize, sfreq=self.samplerate)
            
        elif psdtype == 'morlet':
            freqs = self.freqs
            power = mne.time_frequency.tfr_array_morlet(data, 
                                            sfreq=self.samplerate, 
                                           freqs=self.freqs)

        power = np.log10(np.abs(power))
        return power, freqs

