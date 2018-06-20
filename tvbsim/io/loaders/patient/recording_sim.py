
import json
import os
import re
from datetime import date 
import mne
import numpy as np
from tvbsim.io.loaders.base import BaseLoader 

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
class RecordingSim(BaseLoader):
    chanlabels = []
    allchanlabels = []

    bad_channels = []
    non_eeg_channels = []

    def __init__(self, json_file, real_json_file=None, config=None):
        super(RecordingSim, self).__init__(config)
        
        self.metadata = self._loadjsonfile(json_file)
        self.json_file = json_file
        self.raw_file = os.path.join(os.path.dirname(self.json_file), 
                                    self.metadata['filename'])
        
        # import in the bad/non-eeg-channels
        if real_json_file is not None:
            realmeta = self._loadjsonfile(real_json_file)
            self.bad_channels = realmeta['bad_channels']
            if 'non_seeg_channels' in realmeta.keys():
                self.non_eeg_channels = realmeta['non_seeg_channels']
            elif 'non_eeg_channels' in realmeta.keys():
                self.non_eeg_channels = realmeta['non_eeg_channels']

        self.is_loaded = False
        self.rawdata = None
        self.samplerate = None
        self.time = None

    def _addlinenoise(self, rawdata):
        self.linefreq = 60
        bandwidth = 4
        numharmonics = 4
        # initialize line noise object
        noise = LineNoise(self.linefreq,
                          bandwidth,
                          numharmonics,
                          self.samplerate)
        numsamps = rawdata.shape[1]

        self.logger.debug(
            "Adding line noise at {} hz with {} harmonics".format(
                self.linefreq, numharmonics))
        # create a copy and add noise to it
        test = rawdata.copy()
        for i in range(rawdata.shape[0]):
            test[i, :] += noise.generate(numsamps)
        rawdata = test
        return rawdata


    def load(self, chanxyzlabels=[], chunkind=None, clip=False, reference='monopolar'):
        if chunkind is not None and clip:
            raise Exception("Can not chunk and clip data! Just choose one")

        if not self.is_loaded:
            datafilepath = self.raw_file
            # extract raw object
            data = np.load(self.datafile, encoding='latin1')
            # extract state variables and source signal
            epits = data['epits']
            zts = data['zts']
            rawdata = data['seegts']
            state_vars = data['state_vars']

            # get events
            if datafilepath.endswith('.edf'):
                events = self.raw.find_edf_events()
                self.events = np.array(events)
                # self.__setevents(self.events)

            # self.rawdata = raw.get_data()[:, :]
            # load in all the data from the info data struct
            self._loadinfodata()
            self.time = (1./self.samplerate) * np.r_[:self.raw.n_times]
            self.chanlabels = self.raw.ch_names
            
            # convert all channel labels to lowercase
            self.chanlabels = np.array([ch.lower() for ch in self.chanlabels])
            self.bad_channels = np.array([ch.lower() for ch in self.bad_channels])
            self.non_eeg_channels = np.array([ch.lower() for ch in self.non_eeg_channels])
            self.chanxyzlabels = np.array([ch.lower() for ch in chanxyzlabels])

            # get the channel masks for bad, non-seeg, and not-in-xyz coords
            self._bad_channel_mask = np.array([ch in self.bad_channels for ch in self.chanlabels], dtype=bool)
            self._non_seeg_channels_mask = np.array([ch in self.non_eeg_channels for ch in self.chanlabels], dtype=bool)
            self._non_xyz_mask = np.array([ch not in self.chanxyzlabels for ch in self.chanlabels], dtype=bool)

            # set reference and compute chunks of data
            self.reference = reference
            self.logger.debug("reference is ", reference)
            if chunkind is not None:
                self.computechunks()

            # extract the data
            if reference == 'monopolar':
                self.logger.info("Loading monopolar data!")
                self.chanlabels = self.get_ch_names_monopolar()
                if not clip and chunkind is not None:
                    rawdata = self.get_data_monopolar(chunkind=chunkind)
                elif clip and self.type == 'interictal':
                    rawdata = self.clipinterictal()
                elif clip and 'seizure' in self.type:
                    rawdata = self.clipseizure()
                else:
                    rawdata = self.get_data_monopolar()
            elif reference == 'bipolar':
                self.logger.info("Loading bipolar data!")
                self.set_bipolar()
                self.chanlabels = self.get_ch_names_bipolar()
                if not clip and chunkind is not None:
                    rawdata = self.get_data_bipolar(chunkind=chunkind)
                elif clip and self.type == 'interictal':
                    rawdata = self.clipinterictal()
                elif clip and 'seizure' in self.type:
                    rawdata = self.clipseizure()
                else:
                    rawdata = self.get_data_bipolar()
            # filter the data
            rawdata = self.filter_data(rawdata)

            # add line noise
            rawdata = self._addlinenoise(rawdata)

            # set metadata now that all rawdata is processed
            self.setmetadata()
            
            return rawdata

    def _loadinfodata(self):
        # set samplerate
        self.samplerate = self.metadata['samplerate']
        # set channel names
        self.chanlabels = self.metadata['chanlabels']
        # else grab it from the json object
        self.onset_ind = self.metadata['onsettimes']
        self.offset_ind = self.metadata['offsettimes']
        self.offset_sec = np.divide(self.offset_ind, self.samplerate)
        self.onset_sec = np.divide(self.onset_ind, self.samplerate)

    def set_bipolar(self):
        self._ch_names_bipolar = []
        self._bipolar_inds = []
        self._bad_channel_mask_bipolar = []

        n = len(self.chanlabels)

        for inds in zip(np.r_[:n-1], np.r_[1:n]):

            if np.any([self._non_seeg_channels_mask[ind] for ind in inds]):
                continue

            names = [self.chanlabels[ind] for ind in inds]

            elec0, num0 = re.match("^([A-Za-z]+[']?)([0-9]+)$", names[0]).groups()
            elec1, num1 = re.match("^([A-Za-z]+[']?)([0-9]+)$", names[1]).groups()

            if elec0 == elec1 and abs(int(num0) - int(num1)) == 1:
                name = "%s%s-%s" % (elec0, num1, num0)
                self._ch_names_bipolar.append(name)
                self._bipolar_inds.append([inds[1], inds[0]])
                self._bad_channel_mask_bipolar.append(np.any(self._bad_channel_mask[list(inds)]))

        self._bipolar_inds = np.array(self._bipolar_inds, dtype=int)
        self._bad_channel_mask_bipolar = np.array(self._bad_channel_mask_bipolar, dtype=bool)

    def _get_channel_mask(self, include_bad, include_non_seeg, include_non_xyz):
        mask = np.ones(len(self.raw.ch_names), dtype=bool)
        if not include_bad:
            mask *= ~ self._bad_channel_mask
        if not include_non_seeg:
            mask *= ~ self._non_seeg_channels_mask
        if not include_non_xyz:
            mask *= ~ self._non_xyz_mask
        return mask

    def get_data_monopolar_old(self, include_bad=False, include_non_seeg=False, include_non_xyz=False,
                             avg_ref=False, chunkind=None):
        # if not self.is_loaded:
        #     return None

        mask = self._get_channel_mask(include_bad, include_non_seeg, include_non_xyz)
        if chunkind is not None:
            win = self.winlist[ind]
            data = self.raw.get_data()[mask, win[0]:win[1] + 1]
        else:
            data = self.raw.get_data()[mask, :]

        if avg_ref:
            data -= np.mean(data, axis=0)
        return data
    def get_data_monopolar(self, chunkind=None):
        if chunkind is not None:
            win = self.winlist[ind]
            data = self.raw.get_data()[:, win[0]:win[1] + 1]
        else:
            data = self.raw.get_data()[:, :]

        return data

    def get_data_bipolar(self, include_bad=False,  chunkind=None):
        # if not self.is_loaded:
        #     return None

        data = self.rawdata[self._bipolar_inds[:, 1], :] - self.rawdata[self._bipolar_inds[:, 0], :]

        if chunkind is not None:
            win = self.winlist[ind]
            data = data[:, win[0]:win[1] + 1]

        if include_bad:
            return data
        else:
            return data[~self._bad_channel_mask_bipolar, :]

    def get_ch_names_monopolar_old(self, include_bad=False, include_non_seeg=False, include_non_xyz=False):
        mask = self._get_channel_mask(include_bad, include_non_seeg, include_non_xyz)
        return list(np.array(self.chanlabels, dtype=str)[mask])

    def get_ch_names_monopolar(self):
        return list(np.array(self.chanlabels, dtype=str))

    def get_ch_names_bipolar(self, include_bad=False):
        if not self.is_loaded:
            return None

        if include_bad:
            return self._ch_names_bipolar
        else:
            return list(np.array(self._ch_names_bipolar, dtype=str)[~ self._bad_channel_mask_bipolar])

    def filter_data(self, rawdata):
        rawdata = np.array(rawdata)
        # the bandpass range to pass initial filtering
        freqrange = [0.5]
        freqrange.append(self.samplerate // 2 - 1)
        # set edge freqs that were used in recording
        # Note: the highpass_freq is the max frequency we should be able to see
        # then.
        self.lowpass_freq = freqrange[0]
        self.highpass_freq = freqrange[1]
        # the notch filter to apply at line freqs
        linefreq = int(self.linefreq)           # LINE NOISE OF HZ
        assert linefreq == 50 or linefreq == 60
        self.logger.debug("Line freq is: %s" % linefreq)
        # initialize the line freq and its harmonics
        freqs = np.arange(linefreq, 251, linefreq)
        freqs = np.delete(freqs, np.where(freqs > self.samplerate // 2)[0])

        rawdata = mne.filter.filter_data(rawdata,
                                         sfreq=self.samplerate,
                                         l_freq=freqrange[0],
                                         h_freq=freqrange[1],
                                         # pad='reflect',
                                         verbose=False
                                         )
        rawdata = mne.filter.notch_filter(rawdata,
                                          Fs=self.samplerate,
                                          freqs=freqs,
                                          verbose=False
                                          )
        return rawdata

    def setmetadata(self):
        """
        If the user wants to clip the data, then you can save a separate metadata
        file that contains all useful metadata about the dataset.
        """
        if not self.is_loaded:
            # Set data from the mne file object
            self.metadata['samplerate'] = self.samplerate
            self.metadata['chanlabels'] = self.chanlabels
            self.metadata['lowpass_freq'] = self.lowpass_freq
            self.metadata['highpass_freq'] = self.highpass_freq
            self.metadata['record_date'] = self.record_date
            self.metadata['linefreq'] = self.linefreq
            self.metadata['onsetsec'] = self.onset_sec
            self.metadata['offsetsec'] = self.offset_sec
            self.metadata['onsetind'] = self.onset_ind
            self.metadata['offsetind'] = self.offset_ind
            self.metadata['rawfilename'] = self.fif_file
            # self.metadata['patient'] = self.patient
            self.metadata['reference'] = self.reference

            self.metadata['bad_channels'] = self.bad_channels
            self.metadata['non_eeg_channels'] = self.non_eeg_channels
     
    def __repr__(self):
        return "Recording('%s')" % os.path.basename(self.json_file)
