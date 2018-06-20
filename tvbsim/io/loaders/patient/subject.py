import os
import zipfile

import numpy as np

from tvbsim.io.config import dataconfig as dataconfig
from tvbsim.io.loaders.base import BaseLoader 
from .contacts import Contacts
from .recording import Recording
from .recording_sim import RecordingSim

ICTAL_TYPES = ['ictal', 'sz', 'seiz', 'seizure']
INTERICTAL_TYPES = ['interictal', 'ii', 'aslp', 'aw']

class Subject(BaseLoader):
    sim_recordings = []
    def __init__(self, name, root_dir=None, atlas=None, preload=True, DEBUG=True, SIM=False, config=None):
        super(Subject, self).__init__(config=config)
        self.name = name

        self.subject_dir = dataconfig.subject_dir
        self.atlas = dataconfig.atlas
        self.root_dir = os.path.join(dataconfig.subject_dir, name)

        if root_dir is not None:
            self.root_dir = root_dir
        if atlas is not None:
            self.atlas = atlas

        # initializations - to find files
        self._init_files()
        # load in connectivity
        self._loadconnectivity()
        # load in ez_hypothesis
        self._loadezhypothesis()
        # load in surface
        self._loadsurface()
        # load in contacts
        # self._loadcontacts()
        self._loadseegxyz()
        self._loadgainmat()
        # map contacts to regions using DWI and T1 Parcellation
        self._mapcontacts_toregs()
        self.tempxyzlabels = self.chanxyzlabels
        if DEBUG:
            self._check_all_files()

        if preload:
            if SIM:
                self.read_sim_eeg()
            else:
                self.read_eeg()

    def read_sim_eeg(self):
        self.sim_files = []

        simdatadir = self.root_dir
        json_files = [filename for filename in os.listdir(simdatadir) \
            if filename.endswith('.json') if not filename.startswith('.')]

        for json_file in json_files:
            recording = RecordingSim(os.path.join(simdatadir, json_file))
            self.sim_recordings.append(recording)
            self.sim_files.append(json_file)

    def read_eeg(self):
        self.seizure_files = []
        self.interictal_files = []

        self.seizure_recordings = []
        self.interictal_recordings = []
        self.stimulation_recordings = []

        eeg_dir = os.path.join(self.root_dir, "seeg", "fif")
        if not self._exists(eeg_dir):
            eeg_dir = os.path.join(self.root_dir, "seeg", "edf")
        if not self._exists(eeg_dir):
            raise IOError("{} for raw eeg data does not exist. Checked fif and edf!".format(eeg_dir))

        json_files = [filename for filename in os.listdir(eeg_dir) if filename.endswith('.json') if not filename.startswith('.')]

        for json_file in json_files:
            json_filepath = os.path.join(eeg_dir, json_file)
            recording = Recording(json_filepath)
            recording_type = recording.type.strip().lower()
            if recording_type == 'interictal' or 'interictal' in recording_type:
                self.interictal_recordings.append(recording)
                self.interictal_files.append(json_filepath)
            elif recording_type == 'stimulated seizure':
                self.stimulation_recordings.append(recording)
            elif recording_type == 'spontaneous seizure' or 'seizure' in recording_type.lower():
                self.seizure_recordings.append(recording)
                self.seizure_files.append(json_filepath)
            elif any(x  in json_file.lower() for x in ICTAL_TYPES):
                self.logger.debug("No recording type, so assuming it is ictal")
                self.seizure_recordings.append(recording)
                self.seizure_files.append(json_filepath)
            elif any(x in json_file.lower() for x in INTERICTAL_TYPES):
                self.interictal_recordings.append(recording)
                self.interictal_files.append(json_filepath)
            # else:
            #     raise ValueError("Unexpected recording type: %s" % recording.type)

    def sync_good_data(self, chanlabels, contact_regs, include_bad=False, include_non_seeg=False, include_white=False):        
        # reject bad channels and non-seeg contacts
        _bad_channel_mask = np.array([ch in self.bad_channels for ch in chanlabels], dtype=bool)
        _non_seeg_channels_mask = np.array([ch in self.non_eeg_channels for ch in chanlabels], dtype=bool)
        mask = np.ones(len(chanlabels), dtype=bool)

        if not include_bad:
            mask *= ~_bad_channel_mask
        if not include_non_seeg:
            mask *= ~ _non_seeg_channels_mask
        if not include_white and len(contact_regs) > 0:
            gray_mask = self.sync_gray_chans(contact_regs)
            mask *= ~ gray_mask
        return mask

    def load_good_chans_inds(self, chanlabels, 
            bad_channels, 
            non_eeg_channels,
            contact_regs):
        '''
        Function for only getting the "good channel indices" for
        data.

        It may be possible that some data elements are missing
        '''
        # get all the masks and apply them here to the TVB generated data
        # just sync up the raw data avail.
        rawdata_mask = np.ones(len(chanlabels), dtype=bool)
        _bad_channel_mask = np.ones(len(chanlabels), dtype=bool)
        _non_seeg_channels_mask = np.ones(len(chanlabels), dtype=bool)
        _gray_channels_mask = np.ones(len(chanlabels), dtype=bool)
        if len(self.chanxyzlabels) > 0:
            # make sure to preload the raw chan xyz labels
            self._loadseegxyz()
            # sync up xyz and rawdata
            rawdata_mask, _ = self.sync_xyz_and_raw(chanlabels, self.chanxyzlabels)
            # reject white matter contacts
            if len(contact_regs) > 0:
                self._mapcontacts_toregs()
                white_matter_chans = np.array([ch for idx, ch in enumerate(self.chanxyzlabels) if self.contact_regs[idx] == -1])
                _gray_channels_mask = np.array([ch in white_matter_chans for ch in chanlabels], dtype=bool)

        # reject bad channels and non-seeg contacts
        _bad_channel_mask = np.array([ch in bad_channels for ch in chanlabels], dtype=bool)
        _non_seeg_channels_mask = np.array([ch in non_eeg_channels for ch in chanlabels], dtype=bool)
        rawdata_mask *= ~_bad_channel_mask
        rawdata_mask *= ~_non_seeg_channels_mask
        rawdata_mask *= ~_gray_channels_mask
        return rawdata_mask

    def load_simdataset(self, index, reference, sync=True):
        pass
        return rawdata, metadata

    def load_allbadchans(self, recording_type='sz'):
        if recording_type == 'sz':
            recordings = self.seizure_recordings
        elif recording_type == 'ii':
            recordings = self.interictal_recordings
        
        all_recordings = self.seizure_recordings
        for i in self.interictal_recordings:
            all_recordings.append(i)

        bad_channels = []
        non_eeg_channels = []
        for recording in all_recordings:
            bad_channels.extend(recording.bad_channels)
            non_eeg_channels.extend(recording.non_eeg_channels)
        bad_channels = list(set(bad_channels))
        non_eeg_channels = list(set(non_eeg_channels))
        return bad_channels, non_eeg_channels

    def load_dataset(self, index, reference='monopolar', chunkind=None, clip=False, recording_type='ictal', sync=True):
        if recording_type == 'ictal':
            recording = self.seizure_recordings[index]
        elif recording_type == 'interictal':
            recording = self.interictal_recordings[index]
        self.logger.debug("Looking at recording {}".format(recording))
        print("Looking at recording {}".format(recording))

        rawdata = recording.load(self.chanxyzlabels, reference=reference, chunkind=chunkind, clip=clip)
        metadata = recording.metadata

        chanlabels = np.array(recording.chanlabels)
        self.bad_channels = np.array(recording.bad_channels)
        self.non_eeg_channels = np.array(recording.non_eeg_channels)
        chanxyzlabels = np.array([ch.lower() for ch in recording.chanxyzlabels])
        chanxyz = self.chanxyz
        contact_regs = self.contact_regs

        if sync and not recording.loaded:
            mask = np.ones(len(chanlabels), dtype=bool)
            # get all the masks and apply them here to the TVB generated data
            # just sync up the raw data avail.
            if len(chanxyzlabels) > 0:
                print("original shapes: ")
                print(len(chanlabels), len(chanxyzlabels), len(contact_regs), chanxyz.shape)
                # print("new mask: ")
                rawdata_mask, xyzdata_mask = self.sync_xyz_and_raw(chanlabels, chanxyzlabels)
                chanlabels = np.array(chanlabels)[rawdata_mask]
                chanxyzlabels = np.array(chanxyzlabels)[xyzdata_mask]
                chanxyz = np.array(chanxyz)[xyzdata_mask,:]
                if len(contact_regs) > 0:
                    contact_regs = np.array(contact_regs)[xyzdata_mask]
                    assert contact_regs.shape[0] == chanlabels.shape[0]
                rawdata = rawdata[rawdata_mask,:]

                assert chanlabels.shape[0] == rawdata.shape[0]
                assert rawdata.shape[0] == chanxyz.shape[0]

            # sync good data as defined by clinician
            if len(contact_regs) > 0:
                mask = self.sync_good_data(chanlabels, contact_regs)
                contact_regs = contact_regs[mask]
                chanlabels = chanlabels[mask]
                assert contact_regs.shape[0] == chanlabels.shape[0]
            else:
                mask = self.sync_good_data(chanlabels, contact_regs, include_white=True)
                chanlabels = chanlabels[mask]

            if len(chanxyzlabels) > 0:
                chanxyz = chanxyz[mask, :]
                assert chanlabels.shape[0] == chanxyz.shape[0]
            rawdata = rawdata[mask, :]
            # self.logger.debug(len(self.chanlabels))
            # self.logger.debug(len(self.chanxyzlabels))
            # self.logger.debug(len(self.contact_regs))
            # self.logger.debug(rawdata.shape)
            assert chanlabels.shape[0] == rawdata.shape[0]

            recording.loaded = True
        # Set data from external text, connectivity, elec files
        if len(self.region_labels) > 0:
            metadata['ez_region'] = self.region_labels[self.ezinds]
        else:
            metadata['ez_region'] = []
        metadata['region_labels'] = self.region_labels
        metadata['chanxyz'] = chanxyz
        metadata['contact_regs'] = contact_regs
        metadata['chanlabels'] = chanlabels

        return rawdata, metadata
