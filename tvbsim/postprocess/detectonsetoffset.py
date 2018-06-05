import os
import numpy as np
import peakdetect as peakdetect
import scipy
from sklearn.preprocessing import StandardScaler
import warnings

from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass', analog=False)
    return b, a


def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, highcut, fs, order=5):
    nyq = 0.5 * fs
    highcut = highcut / nyq
    b, a = butter(order, highcut, btype='lowpass', analog=False)

    y = scipy.signal.filtfilt(b, a, data)
    return y


class DetectShift(object):
    '''
    A class wrapper for postprocessor of the TVB simulations.

    We want to be able to trim the time series if needed.
    '''

    def getonsetsoffsets(self, epits, allinds):
        highcut = 1
        fs = 1000

        # only look at the source signals with all indices
        # NEED TO CHANGE LATER TO INCLUDE ALL INDICES SINCE SEIZURES CAN SPREAD
        seiz_epi = epits[allinds, :]

        for ireg in range(seiz_epi.shape[0]):
            seiz_epi[ireg, :] = butter_lowpass_filter(
                np.ravel(seiz_epi[ireg, :]), highcut, fs, order=5)

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
