import os
import zipfile

import numpy as np

from tvbsim.io.config import dataconfig as dataconfig
from tvbsim.io.patient.base import BaseSubjectLoader
from .contacts import Contacts
from .recording import Recording

ICTAL_TYPES = ['ictal', 'sz', 'seiz', 'seizure']
INTERICTAL_TYPES = ['interictal', 'ii', 'aslp', 'aw']

class Subject(BaseSubjectLoader):
    sim_recordings = []
    def __init__(self, name, root_dir=None, atlas=None, DEBUG=True, preload=True, config=None):
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
        if self._exists(self.seegfile):
            self._loadcontacts()
            self._loadseegxyz()
        # map contacts to regions using DWI and T1 Parcellation
        if self._exists(self.label_volume_file):
            self._mapcontacts_toregs()

        if DEBUG:
            self._check_all_files()

        if preload:
            # self.read_eeg()
            self.read_sim_eeg()

    def read_sim_eeg(self):
        simdatadir = self.root_dir
        json_files = [filename for filename in os.listdir(simdatadir) \
            if filename.endswith('.json') if not filename.startswith('.')]

        for json_file in json_files:
            recording = RecordingSim(os.path.join(eeg_dir, json_file))
            self.sim_recordings.append(recording)

    def read_eeg(self):
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
            recording = Recording(os.path.join(eeg_dir, json_file))
            recording_type = recording.type.strip().lower()
            if recording_type == 'interictal' or 'interictal' in recording_type:
                self.interictal_recordings.append(recording)
            elif recording_type == 'stimulated seizure':
                self.stimulation_recordings.append(recording)
            elif recording_type == 'spontaneous seizure' or 'seizure' in recording_type.lower():
                self.seizure_recordings.append(recording)
            elif any(x  in json_file.lower() for x in ICTAL_TYPES):
                print("No recording type, so assuming it is ictal")
                self.seizure_recordings.append(recording)
            elif any(x in json_file.lower() for x in INTERICTAL_TYPES):
                self.interictal_recordings.append(recording)
            # else:
            #     raise ValueError("Unexpected recording type: %s" % recording.type)

    def sync_good_data(self, include_bad=False, include_non_seeg=False, include_white=False):        
        # reject bad channels and non-seeg contacts
        _bad_channel_mask = np.array([ch in self.bad_channels for ch in self.chanlabels], dtype=bool)
        _non_seeg_channels_mask = np.array([ch in self.non_eeg_channels for ch in self.chanlabels], dtype=bool)
        mask = np.ones(len(self.chanlabels), dtype=bool)

        if not include_bad:
            mask *= ~_bad_channel_mask
        if not include_non_seeg:
            mask *= ~ _non_seeg_channels_mask
        if not include_white and len(self.contact_regs) > 0:
            gray_mask = self.sync_gray_chans()
            mask *= ~ gray_mask
        return mask

    def load_good_chans_inds(self, chanlabels, bad_channels=[], non_eeg_channels=[]):
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
            if len(self.contact_regs) > 0:
                self._mapcontacts_toregs()
                white_matter_chans = np.array([ch for idx, ch in enumerate(self.chanxyzlabels) if self.contact_regs[idx] == -1])
                _gray_channels_mask = np.array([ch in white_matter_chans for ch in chanlabels], dtype=bool)

        # reject bad channels and non-seeg contacts
        if bad_channels:
            _bad_channel_mask = np.array([ch in bad_channels for ch in chanlabels], dtype=bool)
        if non_eeg_channels:
            _non_seeg_channels_mask = np.array([ch in non_eeg_channels for ch in chanlabels], dtype=bool)
        rawdata_mask *= ~_bad_channel_mask
        rawdata_mask *= ~_non_seeg_channels_mask
        rawdata_mask *= ~_gray_channels_mask
        return rawdata_mask

    def load_dataset(self, index, reference='monopolar', chunkind=None, clip=False, sync=True):
        recording = self.seizure_recordings[index]
        rawdata = recording.load(self.chanxyzlabels, reference=reference, chunkind=chunkind, clip=clip)
        metadata = recording.metadata
        self.chanlabels = np.array(recording.chanlabels)
        self.bad_channels = np.array(recording.bad_channels)
        self.non_eeg_channels = np.array(recording.non_eeg_channels)
        self.chanxyzlabels = np.array([ch.lower() for ch in self.chanxyzlabels])

        if sync:
            mask = np.ones(len(self.chanlabels), dtype=bool)
            # get all the masks and apply them here to the TVB generated data
            # just sync up the raw data avail.
            if len(self.chanxyzlabels) > 0:
                rawdata_mask, xyzdata_mask = self.sync_xyz_and_raw(self.chanlabels, self.chanxyzlabels)
                self.chanlabels = np.array(self.chanlabels)[rawdata_mask]
                self.chanxyzlabels = np.array(self.chanxyzlabels)[xyzdata_mask]
                self.chanxyz = np.array(self.chanxyz)[xyzdata_mask,:]
                if len(self.contact_regs) > 0:
                    self.contact_regs = np.array(self.contact_regs)[xyzdata_mask]
                    assert self.contact_regs.shape[0] == self.chanlabels.shape[0]
                rawdata = rawdata[rawdata_mask,:]

                assert self.chanlabels.shape[0] == rawdata.shape[0]
                assert rawdata.shape[0] == self.chanxyz.shape[0]

            # sync good data as defined by clinician
            if len(self.contact_regs) > 0:
                mask = self.sync_good_data()
                self.contact_regs = self.contact_regs[mask]
                self.chanlabels = self.chanlabels[mask]
                assert self.contact_regs.shape[0] == self.chanlabels.shape[0]
            else:
                mask = self.sync_good_data(include_white=True)
                self.chanlabels = self.chanlabels[mask]



            if len(self.chanxyzlabels) > 0:
                self.chanxyz = self.chanxyz[mask, :]
                assert self.chanlabels.shape[0] == self.chanxyz.shape[0]
            rawdata = rawdata[mask, :]
            # print(len(self.chanlabels))
            # print(len(self.chanxyzlabels))
            # print(len(self.contact_regs))
            # print(rawdata.shape)
            assert self.chanlabels.shape[0] == rawdata.shape[0]

        # Set data from external text, connectivity, elec files
        if len(self.region_labels) > 0:
            metadata['ez_region'] = self.region_labels[self.ezinds]
        else:
            metadata['ez_region'] = []
        metadata['region_labels'] = self.region_labels
        metadata['chanxyz'] = self.chanxyz
        metadata['contact_regs'] = self.contact_regs
        metadata['chanlabels'] = self.chanlabels

        return rawdata, metadata
