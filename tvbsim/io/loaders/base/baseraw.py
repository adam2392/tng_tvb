import numpy as np 
import os
import mne

from tvbsim.io.loaders.base.basemeta import BaseMetaLoader

"""
Usage:

loader.loadraw_jsonfile()
loader.loadraw_datafile()
loader.create_data_masks()
"""

class BaseRawLoader(BaseMetaLoader):
    ezelecs = None
    resectelecs = None
    def __init__(self, root_dir, config):
        super(BaseRawLoader, self).__init__(root_dir=root_dir, 
                                            config=config)
        self.record_filename = None 

        self.onsetsec = None
        self.offsetsec = None
        self.onsetind = None
        self.offsetind = None 

    def loadraw_datafile(self, rawfilepath):
        # if json file is already loaded, assert that the rawdata file to be loaded
        # is correct and is for this json file
        if self.record_filename is not None:
            self.logger.debug("Record filename is {} ".format(self.record_filename))
            self.logger.debug("Rawfilepath is {}".format(rawfilepath))
            assert self.record_filename in rawfilepath

        # extract raw object
        if rawfilepath.endswith('.edf'):
            self.logger.debug("reading edf from {}".format(rawfilepath))
            raw = mne.io.read_raw_edf(rawfilepath, preload=True, verbose='ERROR')
        elif rawfilepath.endswith('.fif'):
            self.logger.debug("reading fif from {}".format(rawfilepath))
            raw = mne.io.read_raw_fif(rawfilepath, preload=False, verbose=False)

        # set samplerate
        self.samplerate = raw.info['sfreq']
        if self.offsetsec is not None:
            self.offsetind = np.multiply(self.offsetsec, self.samplerate)
        else:
            self.offsetind = None
        if self.onsetsec is not None:
            self.onsetind = np.multiply(self.onsetsec, self.samplerate)
        else:
            self.onsetind = None

        self.chanlabels = np.array(self.scrubchannels(raw.ch_names))
        self.mneraw = raw 

    def loadraw_jsonfile(self, jsonfilepath):
        metadata = self._loadjsonfile(jsonfilepath)

        # extract relevant time markers on each patient
        if metadata['onset']:
            self.onsetsec = float(metadata['onset'])
        else:
            self.onsetsec = None
        if metadata['termination']:
            self.offsetsec = float(metadata['termination'])
        else:
            self.offsetsec = None
            
        self.type = metadata['type']
        self.bad_channels = metadata['bad_channels']
        if 'non_seeg_channels' in metadata.keys():
            self.non_eeg_channels = metadata['non_seeg_channels']
        elif 'non_eeg_channels' in metadata.keys():
            self.non_eeg_channels = metadata['non_eeg_channels']
        
        # scrub channels
        self.bad_channels = np.array(self.scrubchannels(self.bad_channels))
        self.non_eeg_channels = np.array(self.scrubchannels(self.non_eeg_channels))

        # store the rawdata filename
        self.record_filename = metadata['filename']

        # inserted code here to try to get ez/resect elecs
        try:
            self.ezelecs = metadata['ez_elecs']
            self.resectelecs = metadata['resect_elecs']
        except:
            self.logger.error("No ez/resect elecs set for this jsonfile")

    def create_data_masks(self):
        # create mask from bad/noneeg channels
        self.badchannelsmask = self.bad_channels
        self.noneegchannelsmask = self.non_eeg_channels

        # create mask from raw recording data and structural data
        if len(self.chanxyzlabels) > 0:
            self.rawdatamask = np.array([ch for ch in self.chanlabels if ch not in self.chanxyzlabels])
        else:
            self.rawdatamask = np.array([])
        if len(self.chanlabels) > 0:
            self.xyzdatamask = np.array([ch for ch in self.chanxyzlabels if ch not in self.chanlabels])
        else:
            self.xyzdatamask = np.array([])

        # print("\n {}, {} \n".format(self.chanlabels, self.chanxyzlabels))
        # print("bad channels: ", self.badchannelsmask)
        # print("noneeg channels: ", self.noneegchannelsmask)
        # print("not in rawdata: ", self.rawdatamask)
        # print("not in xyz data: ", self.xyzdatamask)
        # print("not gray matter contacts: ", self.whitemattermask)
        # self.mask = [
        #     self.badchannelsmask,
        #     self.noneegchannelsmask,
        #     self.whitemattermask
        # ]
    ''' 
    7/4/18: Decided it is easier to debug when data masks are labels of "to not include channels"
    '''
    # def create_data_masks_old(self):
    #     # create mask from bad/noneeg channels
    #     self.badchannelsmask = self.getmask(self.chanlabels, self.bad_channels)
    #     self.noneegchannelsmask = self.getmask(self.chanlabels, self.non_eeg_channels)

    #     # create mask from raw recording data and structural data
    #     self.rawdatamask, self.xyzdatamask = self.sync_xyz_and_raw(self.chanlabels, self.chanxyzlabels)

    # def getmask(self, labels, bad_channels):
    #     _bad_channel_mask = np.array([ch in bad_channels for ch in labels], dtype=bool)
    #     return _bad_channel_mask
    
    # def sync_xyz_and_raw(self, chanlabels, chanxyzlabels):
    #     '''             REJECT BAD CHANS LABELED BY CLINICIAN       '''
    #     # only deal with contacts both in raw data and with xyz coords
    #     _xyz_mask_forraw = np.array([ch in chanxyzlabels for ch in chanlabels], dtype=bool)
    #     _raw_mask_forxyz = np.array([ch in chanlabels for ch in chanxyzlabels], dtype=bool)

    #     if len(chanxyzlabels) == 0:
    #         _xyz_mask_forraw = np.ones(len(chanlabels), dtype=bool)
    #         _raw_mask_forxyz = np.ones(len(chanxyzlabels), dtype=bool)

    #     return _xyz_mask_forraw, _raw_mask_forxyz
