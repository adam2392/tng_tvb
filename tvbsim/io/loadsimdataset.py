import os
import numpy as np
import pandas as pd
import mne
import json
import io

from tvbsim.io.utils import seegrecording
from tvbsim.io.base import BaseLoader
from tvbsim.base.preprocess.util.noise import LineNoise
from datetime import date


class LoadSimDataset(BaseLoader):
    root_dir = None
    rawdatadir = None
    datafile = None
    patient = None

    def __init__(self, root_dir=None, datafile=None, rawdatadir=None, patient=None,
                 preload=False, reference='monopolar', config=None):
        super(LoadSimDataset, self).__init__(config=config)
        self.root_dir = root_dir
        self.rawdatadir = rawdatadir
        self.datafile = datafile
        self.reference = reference
        self.patient = patient

        if patient is not None:
            if patient not in self.rawdatadir:
                self.rawdatadir = os.path.join(self.rawdatadir, patient)

        # set directories for the datasets
        self.seegdir = os.path.join(self.rawdatadir, 'seeg', 'fif')
        self.elecdir = os.path.join(self.rawdatadir, 'elec')
        self.dwidir = os.path.join(self.rawdatadir, 'dwi')
        self.tvbdir = os.path.join(self.rawdatadir, 'tvb')

        # load in the meta data and create mne raw object
        self.loadmeta_tvbdata()

        # preload the processed data
        if preload:
            metafile = datafile.split('.npz')[0] + '.json'
            self._loadjsonfile(metafile)
            self._loadmetadata()
            rawdata = self.loadsimdata()
            self.create_info_obj()
            self.create_raw_obj(rawdata)
            self.load_data()

    def _loadmetadata(self):
        # set line frequency and add to it
        self.samplerate = self.metadata['samplerate']
        self.chanlabels = self.metadata['chanlabels']
        # self.locations = self.metadata['seeg_xyz']

        # else grab it from the json object
        self.onset_ind = self.metadata['onsettimes']
        self.offset_ind = self.metadata['offsettimes']
        self.offset_sec = np.divide(self.offset_ind, self.samplerate)
        self.onset_sec = np.divide(self.onset_ind, self.samplerate)

    def savejsondata(self, metadata, metafilename):
        # save the timepoints, included channels used, parameters
        self._writejsonfile(metadata, metafilename)
        self.logger.info('Saved metadata as json!')
        
    def loadmeta_tvbdata(self):
        self.logger.debug('Reading in metadata!')
        # rename files from .xyz -> .txt
        self._renamefiles()
        self._loadseegxyz()
        self._mapcontacts_toregs()

        # load in ez hypothesis and connectivity from TVB pipeline
        self._loadezhypothesis()
        self._loadconnectivity()
        self.logger.debug("Finished reading in metadata!")

    def load_data(self):
        # run main loader
        self._loadinfodata()
        self.filter_data()
        self.referencedata()
        self.sync_good_data()

    def create_raw_obj(self, data):
        self.raw = mne.io.RawArray(data, self.info)

    def create_info_obj(self):
        sfreq = self.samplerate
        ch_names = list(self.chanlabels)
        linefreq = self.linefreq
        lowpass_freq = 0.1
        highpass_freq = 499
        ch_types = ['seeg'] * len(ch_names)
        # It is also possible to use info from another raw object.
        self.info = mne.create_info(ch_names=ch_names,
                                    sfreq=sfreq,
                                    ch_types=ch_types)
        self.info['line_freq'] = linefreq
        self.info['lowpass'] = lowpass_freq
        self.info['highpass'] = highpass_freq

    def loadsimdata(self):
        """
        Loads in the simulated datasets.

        Simdata includes:
        - epits, seegts, state_vars
        - metadata with samplerate, winsize, chanlabels, chanxyz
        """
        if not self.datafile.startswith(self.root_dir):
            self.datafile = os.path.join(self.root_dir, self.datafile)
        # load in the simulated data from tvb forward simulator
        data = np.load(self.datafile, encoding='latin1')

        # extract state variables and source signal
        self.epits = data['epits']
        self.zts = data['zts']

        # extract SEEG forward solution
        rawdata = data['seegts']

        # add line noise
        rawdata = self._addlinenoise(rawdata)
        return rawdata

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

    def _loadinfodata(self):
        # set samplerate
        self.samplerate = self.raw.info['sfreq']
        # set channel names
        self.chanlabels = self.raw.info['ch_names']
        # also set to all the channel labels
        self.allchans = self.chanlabels
        self._processchanlabels()
        # set edge freqs that were used in recording
        # Note: the highpass_freq is the max frequency we should be able to see
        # then.
        self.lowpass_freq = self.raw.info['lowpass']
        self.highpass_freq = self.raw.info['highpass']
        # set recording date
        record_date = date.fromtimestamp(self.raw.info["meas_date"][0])
        self.record_date = record_date
        # number of microseconds
        record_ms_date = self.raw.info["meas_date"][1]
        self.record_ms_date = record_ms_date
        # set line freq
        self.linefreq = self.raw.info['line_freq']

    def _processchanlabels(self):
        self.chanlabels = [str(x).replace('POL', '').replace(' ', '')
                           for x in self.chanlabels]

    def sync_good_data(self, rawdata=None):
        if rawdata is None:
            # rawdata = self.rawdata
            self.rawdata, self.times = self.raw.get_data(return_times=True)

        # map to lower case for all labels
        self.chanlabels = np.array([lab.lower() for lab in self.chanlabels])

        # reject white matter contacts
        graychans_inds = np.where(self.contact_regs != -1)[0]
        self.contact_regs = self.contact_regs[graychans_inds]
        self.rawdata = self.rawdata[graychans_inds, :]
        self.chanxyz = self.chanxyz[graychans_inds, :]
        self.chanlabels = self.chanlabels[graychans_inds]

        # print(self.contact_regs.shape)
        # print(self.rawdata.shape)
        # print(self.chanxyz.shape)
        assert self.contact_regs.shape[0] == self.chanlabels.shape[0]
        assert self.chanlabels.shape[0] == self.rawdata.shape[0]
        assert self.rawdata.shape[0] == self.chanxyz.shape[0]

    def filter_data(self, rawdata=None):
        # the bandpass range to pass initial filtering
        freqrange = [0.5]
        freqrange.append(self.samplerate // 2 - 1)
        # the notch filter to apply at line freqs
        linefreq = int(self.linefreq)           # LINE NOISE OF HZ
        assert linefreq == 50 or linefreq == 60
        self.logger.debug("Line freq is: %s" % linefreq)
        # initialize the line freq and its harmonics
        freqs = np.arange(linefreq, 251, linefreq)
        freqs = np.delete(freqs, np.where(freqs > self.samplerate // 2)[0])

        if rawdata is None:
            # apply band pass filter
            self.raw.filter(l_freq=freqrange[0],
                            h_freq=freqrange[1])
            # apply the notch filter
            self.raw.notch_filter(freqs=freqs)
        else:
            # print('Filtering!', freqrange)
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

    def referencedata(self):
        # apply referencing to data and channel labels (e.g. monopolar,
        # average, bipolar)
        self.raw = self.raw.load_data()

        if self.reference == 'average':
            self.logger.info('\nUsing average referencing!\n')
            self.raw.set_eeg_reference(
                ref_channels="average", projection=False)
        elif self.reference == 'monopolar':
            self.logger.info('\nUsing monopolar referencing!\n')
            self.raw.set_eeg_reference(ref_channels=[], projection=False)
        elif self.reference == 'bipolar':
            self.logger.info("\nUsing bipolar referencing!\n")
            self.logger.debug("NEED TO CALL ALL PREPROCESSING FUNCTIONS")
            assert 1 == 0
            # convert contacts into a list of tuples as data structure
            contacts = []
            for contact in self.chanlabels:
                thiscontactset = False
                for idx, s in enumerate(contact):
                    if s.isdigit() and not thiscontactset:
                        elec_label = contact[0:idx]
                        thiscontactset = True
                contacts.append((elec_label, int(contact[len(elec_label):])))

            self.rawdata, self.times = self.raw.get_data()
            # compute the bipolar scheme
            recording = util.seegrecording.SeegRecording(
                contacts, self.rawdata, self.samplerate)
            self.chanlabels = np.asarray(recording.get_channel_names_bipolar())
            self.rawdata = recording.get_data_bipolar()

    def getmetadata(self):
        """
        If the user wants to clip the data, then you can save a separate metadata
        file that contains all useful metadata about the dataset.
        """
        metadata = dict()
        # Set data from the mne file object
        metadata['samplerate'] = self.samplerate
        metadata['chanlabels'] = self.chanlabels
        metadata['lowpass_freq'] = self.lowpass_freq
        metadata['highpass_freq'] = self.highpass_freq
        metadata['record_date'] = self.record_date
        metadata['onsetsec'] = self.onset_sec
        metadata['offsetsec'] = self.offset_sec
        metadata['reference'] = self.reference
        metadata['linefreq'] = self.linefreq
        metadata['onsetind'] = self.onset_ind
        metadata['offsetind'] = self.offset_ind
        metadata['allchans'] = self.allchans
        metadata['rawfilename'] = self.datafile
        metadata['patient'] = self.patient

        # Set data from external text, connectivity, elec files
        metadata['ez_region'] = self.conn.region_labels[self.ezinds]
        metadata['region_labels'] = self.conn.region_labels
        # metadata['ez_chans'] = self.contact_regs == metadata['ez_region']
        metadata['chanxyz'] = self.chanxyz
        metadata['contact_regs'] = self.contact_regs

        # add to the metadata structure, the simulation settings for x0!
        metadata['sim_ez_reg'] = self.metadata['ezregs']
        metadata['sim_pz_reg'] = self.metadata['pzregs']
        metadata['sim_x0_norm'] = self.metadata['x0norm']
        metadata['sim_x0_ez'] = self.metadata['x0ez']
        metadata['sim_x0_pz'] = self.metadata['x0pz']

        try:
            metadata['chunklist'] = self.winlist
            metadata['secsperchunk'] = self.secsperchunk
        except BaseException:
            self.logger.info(
                'chunking not set for %s %s \n' %
                (self.patient, self.datafile))

        # needs to be gotten from sync_data()
        # metadata['goodchans'] = dataloader.goodchans
        # metadata['graychans'] = dataloader.graychans_inds

        return metadata
