import os
import numpy as np
import peakdetect as peakdetect

class PostProcessor(object):
    '''
    A class wrapper for postprocessor of the TVB simulations.

    We want to be able to trim the time series if needed.
    '''
    def __init__(self, samplerate, allszindices):
        self.samplerate = samplerate
        self.allindices = allszindices

    def postprocts(self, epits, seegts, times, secstoreject=15):
        # reject certain 5 seconds of simulation
        sampstoreject = secstoreject * self.samplerate

        # get the time series processed and squeezed that we want to save
        new_times = times[sampstoreject:]
        new_epits = epits[sampstoreject:, 1, :, :].squeeze().T
        new_zts = epits[sampstoreject:, 0, :, :].squeeze().T
        new_seegts = seegts[sampstoreject:, :, :, :].squeeze().T
        return new_times, new_epits, new_seegts, new_zts

    # assuming onset is the first bifurcation and then every other one is onsets
    # every other bifurcation after the first one is the offset
    def _findonsetoffset(self, signal, delta=0.2/8):
        # get list of tuples for offset, onset respectively
        maxpeaks, minpeaks = peakdetect.peakdetect(signal.squeeze(), delta=delta)
        # store the number detected
        numonsets = len(minpeaks)
        numoffsets = len(maxpeaks)

        onsettimes = []
        offsettimes = []
        # only get the positions the peaks occur at
        for i in range(0,numonsets):
            onsettimes.append(minpeaks[i][0])
        for i in range(0,numoffsets):
            offsettimes.append(maxpeaks[i][0])

        # pad the arrays to have nans if the array sizes are uneven
        if numonsets > numoffsets:
            offsettimes.append(np.nan)
        elif numonsets < numoffsets:
            onsettimes.append(np.nan)

        # convert to numpy arrays and return
        onsettimes = np.array(onsettimes)
        offsettimes = np.array(offsettimes)
        return onsettimes, offsettimes

    def getonsetsoffsets(self, zts, indices,delta=0.2/8):
        # create lambda function for checking the indices
        # check = lambda indices: isinstance(indices,np.ndarray) and len(indices)>=1
        settimes = []

        # go through and get onset/offset times of ez indices
        # if check(indices):
        for index in np.asarray(indices):
            _onsettimes, _offsettimes = self._findonsetoffset(zts[index, :].squeeze(), delta=delta)
            settimes.append(list(zip(_onsettimes, _offsettimes)))
                
        # flatten out list structure if there is one
        settimes = [item for sublist in settimes for item in sublist]
        settimes = np.asarray(settimes).squeeze()

        # do an error check and reshape arrays if necessary
        if settimes.ndim == 1:
            settimes = settimes.reshape(1,settimes.shape[0])
        return settimes

    def getseiztimes(self, settimes):
        # perform some checks
        if settimes.size == 0:
            print("no onset/offset available!")
            return 0

        # sort in place the settimes by onsets, since those will forsure have 1
        settimes = settimes[settimes[:,0].argsort()]

        # get the onsets/offset pairs now
        onsettimes = settimes[:,0]
        offsettimes = settimes[:,1]
        seizonsets = []
        seizoffsets = []
        
        # start loop after the first onset/offset pair
        for i in range(0,len(onsettimes)):        
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
                if curronset > prevoffset:
                    seizoffsets.append(prevoffset)
                    prevonset = curronset
                    prevoffset = curroffset
                    seizonsets.append(prevonset)
                else:
                    # just move the offset along
                    prevoffset = curroffset
            # if at any point, offset is nan, then just return
            if np.isnan(prevoffset):
                print('returning cuz prevoffset is nan!')
                return seizonsets, seizoffsets

        if not np.isnan(prevoffset):
            seizoffsets.append(prevoffset)

        return seizonsets, seizoffsets
