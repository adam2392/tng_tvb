import os
import numpy as np
import peakdetect as peakdetect
import scipy
from sklearn.preprocessing import StandardScaler
import warnings


class PostProcessor(object):
    '''
    A class wrapper for postprocessor of the TVB simulations.

    We want to be able to trim the time series if needed.
    '''

    def __init__(self, samplerate, allszindices):
        self.samplerate = samplerate
        self.allindices = allszindices

    def postprocts(self, statevars_ts, seegts, times, secstoreject=15):
        # reject certain 5 seconds of simulation
        sampstoreject = secstoreject * self.samplerate

        # get the time series processed and squeezed that we want to save
        new_times = times[sampstoreject:]
        new_epits = statevars_ts[sampstoreject:, 1, :, :].squeeze().T
        new_zts = statevars_ts[sampstoreject:, 0, :, :].squeeze().T
        new_seegts = seegts[sampstoreject:, :, :, :].squeeze().T

        # get the other state variables
        x1 = statevars_ts[sampstoreject:, 2, :, :].squeeze().T
        x2 = statevars_ts[sampstoreject:, 3, :, :].squeeze().T
        y1 = statevars_ts[sampstoreject:, 4, :, :].squeeze().T
        y2 = statevars_ts[sampstoreject:, 5, :, :].squeeze().T
        g = statevars_ts[sampstoreject:, 6, :, :].squeeze().T
        state_vars = {
            'x1': x1,
            'x2': x2,
            'y1': y1,
            'y2': y2,
            'g': g
        }

        return new_times, new_epits, new_seegts, new_zts, state_vars

    def postprocts_2(self, epits, seegts, zts, times, secstoreject=0):
        # reject certain 5 seconds of simulation
        sampstoreject = secstoreject * self.samplerate

        # get the time series processed and squeezed that we want to save
        new_times = times[sampstoreject:]
        new_epits = epits[:, sampstoreject:]
        new_zts = zts[:, sampstoreject:]
        new_seegts = seegts[:, sampstoreject:]
        return new_times, new_epits, new_seegts, new_zts

    # assuming onset is the first bifurcation and then every other one is onsets
    # every other bifurcation after the first one is the offset
    def _findonsetoffset(self, signal, lookahead=500, delta=0.2 / 8):
        '''
        Function that uses the peakdetect algorithm with lookahead
        to get the onsets and offsets

        Returns two lists that can be empty if no onset/offsets are found
        '''
        # get list of tuples for offset, onset respectively
        maxpeaks, minpeaks = peakdetect.peakdetect(
            signal.squeeze(), lookahead=lookahead, delta=delta)
        # store the number detected
        numonsets = len(minpeaks)
        numoffsets = len(maxpeaks)

        onsettimes = []
        offsettimes = []
        # only get the positions the peaks occur at
        for i in range(0, numonsets):
            onsettimes.append(minpeaks[i][0])
        for i in range(0, numoffsets):
            offsettimes.append(maxpeaks[i][0])

        # pad the arrays to have nans if the array sizes are uneven
        if numonsets > numoffsets:
            offsettimes.append(np.nan)
        elif numonsets < numoffsets:
            onsettimes.append(np.nan)

        return onsettimes, offsettimes
        # convert to numpy arrays and return
        onsettimes = np.array(onsettimes)
        offsettimes = np.array(offsettimes)
        return onsettimes, offsettimes

    def _normalize(self, zts):
        # apply z normalization
        zts = (zts - np.mean(zts, axis=-1, keepdims=True)) / \
            np.std(zts, axis=-1, keepdims=True)
        return zts

    def _scale(self, zts):
        pass

    def getonsetsoffsets(self, epits, allinds):
        seiz_epi = epits[allinds, :]

        seizonsets = []
        seizoffsets = []
        for ind in range(len(allinds)):
            curr_epi = seiz_epi[ind, :].squeeze()

            # initialize pointer
            pointer = 0
            while pointer < len(curr_epi):
                # look ahead - onset
                #             minind = np.where(curr_epi[pointer:] < np.ceil(np.mean(np.min(seiz_epi, axis=1))))[0]
                minind = np.where(curr_epi[pointer:] < -0.5)[0]

                if len(minind) > 0:
                    seizonsets.append(minind[0] + pointer)
                    # update pointer
                    pointer += minind[0]
                    # look ahead - offset
        #             maxind = np.where(curr_epi[pointer:] > np.ceil(np.mean(np.mean(seiz_epi,axis=1))))[0]
                    maxind = np.where(curr_epi[pointer:] > 0)[0]

                    if len(maxind) > 0:
                        seizoffsets.append(maxind[0] + pointer)
                        # update pointer
                        pointer += maxind[0]
                    else:
                        seizoffsets.append(np.nan)
                        pointer = len(curr_epi)
                else:
                    pointer = len(curr_epi)

        # get the settimes by putting together onsets/offset found
        settimes = np.vstack((seizonsets, seizoffsets)).T
        # sort in place the settimes by onsets, since those will forsure have 1
        settimes = settimes[settimes[:, 0].argsort()]
        return settimes

    def _old_getonsetsoffsets(
            self, zts, indices, lookahead=500, delta=0.2 / 8):
        # assert zts.ndim == 2
        buffzts = zts
        # apply z normalization
        zts = (zts - np.mean(zts, axis=-1, keepdims=True)) / \
            np.std(zts, axis=-1, keepdims=True)

        # create list of the tuple times to store
        settimes = []

        for index in np.asarray(indices):
            currentz = zts[index, :].squeeze()
            minsig = np.min(currentz.ravel())
            currentz[abs(currentz) < abs(currentz * 0.9)] = 0

            # HARD CODED THRESHOLD ON THE RANGE OF THE Z VALUES in this region
            # ensures we don't try to find peaks if there are none
            if np.ptp(buffzts[index, :]) > 0.3:
                _onsettimes, _offsettimes = self._findonsetoffset(currentz,
                                                                  lookahead=lookahead,
                                                                  delta=delta)
                settimes.append(list(zip(_onsettimes, _offsettimes)))
            else:
                print('Skipping this index %d' % index)
        # flatten out list structure if there is one
        settimes = [item for sublist in settimes for item in sublist]
        settimes = np.asarray(settimes).squeeze()

        # do an error check and reshape arrays if necessary
        if settimes.ndim == 1:
            settimes = settimes.reshape(1, settimes.shape[0])

        # sort in place the settimes by onsets, since those will forsure have 1
        try:
            settimes = settimes[settimes[:, 0].argsort()]
        except IndexError:
            warnings.warn('Probably no settimes detected for this patient.'
                          'Need to reanalyze z tiem series.')
            settimes = settimes

        return settimes

    def getonsetsoffsets_new(self, zts, indices, lookahead=500, delta=0.2 / 8):
        # create lambda function for checking the indices
        settimes = []

        # go through and get onset/offset times of ez indices
        for index in np.asarray(indices):
            signal = zts[index, :].squeeze()
            signal = self.processz(signal)

            # GET THE MAX/MIN peaks
            _offsettimes, _onsettimes = self._findonsetoffset(signal,
                                                              lookahead=lookahead,
                                                              delta=delta)
            settimes.append(list(zip(_onsettimes, _offsettimes)))

        # flatten out list structure if there is one
        settimes = [item for sublist in settimes for item in sublist]
        settimes = np.asarray(settimes)

        print(settimes)
        # do an error check and reshape arrays if necessary
        if settimes.ndim == 1:
            settimes = settimes.reshape(1, settimes.shape[0])

        # sort in place the settimes by onsets, since those will forsure have 1
        try:
            settimes = settimes[settimes[:, 0].argsort()]
        except IndexError:
            warnings.warn(
                'Probably no settimes detected for this patient. Need to reanalyze z tiem series.')
            settimes = settimes

        return settimes

    def processz(self, signal):
        # this is a threshold on the differenced z signal - set based on range
        # of z
        threshold = 1
        # low pass filtering parameters
        nyq = self.samplerate / 2.
        order = 5
        cut = 1
        Wn = cut / nyq

        signal = np.diff(signal, n=1)
        signal = StandardScaler().fit_transform(signal[:, np.newaxis])
        # print(signal.shape)
        # APPLY LOW PASS FILTERING
        b, a = scipy.signal.butter(
            N=order, Wn=cut / nyq, btype='low', analog=False)
        # run filtfilt for zero phase distortion
        signal = scipy.signal.filtfilt(b, a, signal.squeeze())

        # APPLY A THRESHOLDING
        signal[abs(signal) < threshold] = 0
        return signal

    @staticmethod
    def getseiztimes(settimes, epsilon=100):
        # perform some checks
        if settimes.size == 0:
            print("no onset/offset available!")
            return [0], [0]

        # sort in place the settimes by onsets, since those will forsure have 1
        settimes = settimes[settimes[:, 0].argsort()]

        # get the onsets/offset pairs now
        onsettimes = settimes[:, 0]
        offsettimes = settimes[:, 1]
        seizonsets = []
        seizoffsets = []

        print(onsettimes)
        print(offsettimes)
        # start loop after the first onset/offset pair
        for i in range(0, len(onsettimes)):
            # get current onset/offset times
            curronset = onsettimes[i]
            curroffset = offsettimes[i]

            # handle first case
            if i == 0:
                prevonset = curronset
                prevoffset = curroffset
                seizonsets.append(prevonset)
            # check current onset/offset
            else:
                # if the onset now is greater then the offset
                # we have one seizure instance
                if curronset > prevoffset + epsilon:
                    seizonsets.append(curronset)
                    seizoffsets.append(prevoffset)
                    prevonset = curronset
                    prevoffset = curroffset

                elif curroffset > prevoffset:
                    prevoffset = curroffset

                elif curroffset < prevoffset:
                    prevoffset = prevoffset

                elif curroffset == np.nan:
                    prevoffset = prevoffset

                else:
                    prevoffset = curroffset
            # if at any point, offset is nan, then just return
            if np.isnan(prevoffset):
                print('returning cuz prevoffset is nan!')
                return seizonsets, seizoffsets

        if np.isnan(prevoffset):
            seizoffsets[-1] = np.nan
        if not np.isnan(prevoffset):
            seizoffsets.append(prevoffset)

        return seizonsets, seizoffsets
