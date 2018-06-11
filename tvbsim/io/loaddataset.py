import os
import numpy as np
import pandas as pd
import mne
import json
import sys

from tvbsim.io.base import BaseLoader
from .utils import seegrecording

from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger
from datetime import date
import warnings

warnings.filterwarnings("ignore", ".*not conform to MNE naming conventions.*")
"""
TODO:
- add this?
        # get relevant channel data - ONLY USED FOR NON TNG PIPELINE DATA
        # if patient is not None:
        #     self.patid, self.seizid = utils.splitpatient(patient)
        #     self.included_chans, self.onsetchans, self.clinresult = utils.returnindices(
        #         self.patid, self.seizid)

        #     # mask bad channels - HARDCODED included chans...
        #     self.maskbadchans(self.included_chans)
"""


class LoadDataset(BaseLoader):
    raw = None
    rawdata = None
    chanlabels = None
    goodchan_labels = None

    contact_regs = None
    conn = None

    onset_sec = None
    onset_ind = None
    offset_sec = None
    offset_ind = None

    def __init__(self, root_dir, datafile, patient=None,
                 preload=False, reference='monopolar', config=None):
        super(LoadDataset, self).__init__(config=config)

        self.root_dir = root_dir
        self.datafile = datafile
        self.reference = reference
        self.patient = patient

        # set directories for the datasets
        self.seegdir = os.path.join(self.root_dir, 'seeg', 'fif')
        if not os.path.exists(self.seegdir):
            self.seegdir = os.path.join(self.root_dir, 'seeg', 'edf')
        self.elecdir = os.path.join(self.root_dir, 'elec')
        self.dwidir = os.path.join(self.root_dir, 'dwi')
        self.tvbdir = os.path.join(self.root_dir, 'tvb')

        self.loadpatientmetadata()
        if preload:
            self.loadrawdata()
            self.filter_data()
            self.sync_good_data()
            
    def loadpatientmetadata(self):
        # sync good data
        # self.sync_good_data()
        self.logger.info('Reading in metadata!')
        # rename files from .xyz -> .txt
        self._renamefiles()

        if os.path.exists(self.elecdir):
            self._loadseegxyz()
        if os.path.exists(self.elecdir) and os.path.exists(self.dwidir):    
            self._mapcontacts_toregs()
        # load in ez hypothesis and connectivity from TVB pipeline
        if os.path.exists(self.tvbdir):
            self._loadezhypothesis()
            self._loadconnectivity()

    def loadrawdata(self):
        # run main loader
        self._loadfile()
        self._loadinfodata()
        # apply referencing to data and channel labels (e.g. monopolar,
        # average, bipolar)
        self.referencedata(FILTER=False)

    def _loadfile(self):
        # load in the json file for this dataset
        metadatafilepath = os.path.join(self.seegdir, self.datafile)
        if metadatafilepath.endswith('.edf'):
            metadatafilepath = metadatafilepath.split('.edf')[0]
        if not metadatafilepath.endswith('.json'):
            metadatafilepath += '.json'
        if not os.path.exists(metadatafilepath):
            self.datafile = self.datafile.split('.json')[0]
            metadatafilepath = os.path.join(self.seegdir, self.datafile+'_0001.json')
        self.logger.debug(
            "The meta data file to use is %s \n" %
            metadatafilepath)
        self._loadjsonfile(metadatafilepath)

        # load in the useful metadata
        rawfile = self.metadata['filename']
        # set if this is a tngpipeline dataset
        datafilepath = os.path.join(self.seegdir, rawfile)
        if not os.path.exists(datafilepath):
            # set if this is a tngpipeline dataset
            datafilepath = os.path.join(self.seegdir, rawfile.lower())

        self.rawfilepath = datafilepath
        # extract raw object
        if datafilepath.endswith('.edf'):
            raw = mne.io.read_raw_edf(datafilepath,
                                      preload=True,
                                      verbose=False)
        elif datafilepath.endswith('.fif'):
            raw = mne.io.read_raw_fif(datafilepath,
                                      # preload=True,
                                      verbose=False)
        else:
            sys.stderr.write("Is this a real dataset? %s \n" % datafilepath)
            print("Is this a real dataset? ", datafilepath)

        # provide loader object access to raw mne object
        self.raw = raw

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
        meas_date = self.raw.info["meas_date"]
        if isinstance(meas_date, list) or isinstance(meas_date, np.ndarray):
            record_date = date.fromtimestamp(meas_date[0])
            # number of microseconds
            record_ms_date = self.raw.info["meas_date"][1]
        elif meas_date is not None:
            record_date = date.fromtimestamp(meas_date)
            record_ms_date = None
        else:
            record_date = None
            record_ms_date = None
        self.record_date = record_date
        self.record_ms_date = record_ms_date

        # set line freq
        self.linefreq = self.raw.info['line_freq']
        if self.linefreq is None:
            self.linefreq = 50
            # self.linefreq = 60
            self.logger.debug(
                "\nHARD SETTING THE LINE FREQ. MAKE SURE TO CHANGE BETWEEN USA/FRANCE DATA!\n")

        # else grab it from the json object
        self.onset_sec = self.metadata['onset']
        self.offset_sec = self.metadata['termination']
        badchans = self.metadata['bad_channels']
        try:
            nonchans = self.metadata['non_seeg_channels']
        except BaseException:
            nonchans = []
        self.badchans = badchans + nonchans
        self.sztype = self.metadata['type']
        if self.offset_sec is not None:
            self.offset_ind = np.multiply(self.offset_sec, self.samplerate)
            self.onset_ind = np.multiply(self.onset_sec, self.samplerate)
        else:
            self.offset_ind = None
            self.onset_ind = None

        # get events
        if self.rawfilepath.endswith('.edf'):
            events = self.raw.find_edf_events()
            self.events = np.array(events)
            if self.events.size > 0:
                self.__setevents(self.events)
        else:
            events = mne.find_events(raw,
                                    stim_channel=[])
            self.events = np.array(events)


    def _processchanlabels(self):
        self.chanlabels = [str(x).replace('POL', '').replace(' ', '')
                           for x in self.chanlabels]

    def __setevents(self, events):
        eventonsets = events[:, 0]
        eventdurations = events[:, 1]
        eventnames = events[:, 2]

        # initialize list of onset/offset seconds
        onset_secs = []
        offset_secs = []
        onsetset = False
        offsetset = False

        # iterate through the events and assign onset/offset times if avail.
        for idx, name in enumerate(eventnames):
            name = name.lower().split(' ')
            if 'onset' in name \
                    or 'crise' in name \
                    or 'cgtc' in name \
                    or 'sz' in name or 'absence' in name:
                if not onsetset:
                    onset_secs = float(eventonsets[idx])
                    onsetset = True
            if 'offset' in name \
                    or 'fin' in name \
                    or 'end' in name:
                if not offsetset:
                    offset_secs = float(eventonsets[idx])
                    offsetset = True

        # set onset/offset times and markers
        try:
            self.onset_sec = onset_secs
            self.onset_ind = np.ceil(onset_secs * int(self.samplerate)).astype(int)
        except TypeError as e:
            print(e)
            self.logger.info("no onset time!")
            self.onset_sec = None
            self.onset_ind = None
        try:
            self.offset_sec = offset_secs
            self.offset_ind = np.ceil(offset_secs * int(self.samplerate)).astype(int)
        except TypeError as e:
            print(e)
            self.logger.info("no offset time!")
            self.offset_sec = None
            self.offset_ind = None

    def sync_good_data(self, rawdata=None):
        if rawdata is None and self.rawdata is None:
            self.rawdata, self.times = self.raw.get_data(return_times=True)
            rawdata = self.rawdata

        badchans = self.badchans
        chanxyz_labs = self.chanxyz_labels
        chanlabels = self.chanlabels
        contact_regs = self.contact_regs
        chanxyz = self.chanxyz

        '''             REJECT BAD CHANS LABELED BY CLINICIAN       '''
        # map badchans, chanlabels to lower case
        badchans = np.array([lab.lower() for lab in badchans])
        chanxyz_labs = np.array([lab.lower() for lab in chanxyz_labs])
        chanlabels = np.array([lab.lower() for lab in chanlabels])

        # extract necessary metadata
        goodchans_inds = [idx for idx, chan in enumerate(
            chanlabels) if chan not in badchans if chan in chanxyz_labs]
        # only grab the good channels specified
        goodchan_labels = chanlabels[goodchans_inds]
        # rawdata = rawdata[goodchans_inds,:]

        '''             GET UNION OF CHANXYZ AND CONTACT LABELS IN RECORDING       '''
        # now sift through our contacts with xyz coords and region_mappings
        reggoodchans_inds = [idx for idx, chan in enumerate(
            chanxyz_labs) if chan in goodchan_labels]
        contact_regs = contact_regs[reggoodchans_inds]
        chanxyz = chanxyz[reggoodchans_inds, :]

        # covert to numpy arrays
        contact_regs = np.array(contact_regs)
        goodchan_labels = np.array(goodchan_labels)

        '''             REJECT WHITE MATTER CONTACTS       '''
        # reject white matter contacts
        graychans_inds = np.where(np.asarray(contact_regs) != -1)[0]
        self.contact_regs = contact_regs[graychans_inds]
        self.rawdata = rawdata[graychans_inds, :]
        self.chanxyz = chanxyz[graychans_inds, :]
        self.chanlabels = goodchan_labels[graychans_inds]

        # print(self.contact_regs.shape)
        # print(self.chanlabels.shape)
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
        # sys.stderr.write("Going to filter at freqrange: \n")
        # sys.stderr.write(freqrange)

        if rawdata is None:
            # apply referencing to data and channel labels (e.g. monopolar,
            # average, bipolar)
            self.raw = self.raw.load_data()

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

    def referencedata(self, FILTER=False):
        # Should we apply bandpass filtering and notch of line noise?
        if FILTER:
            self.filter_data()

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

            self.rawdata, self.times = self.raw.get_data(return_times=True)
            # compute the bipolar scheme
            recording = util.seegrecording.SeegRecording(
                contacts, self.rawdata, self.samplerate)
            self.chanlabels = np.asarray(recording.get_channel_names_bipolar())
            self.rawdata = recording.get_data_bipolar()

    def maskbadchans(self, included_chans=None):
        if self.reference == 'bipolar':
            warnings.warn(
                'Bipolar referencing could not work with hard coded included chans!')

        if included_chans is None:
            warnings.warn('Included chans is hardcoded as: NONE')
            self.logger.info('Doing nothing in maskbadchans')
        else:
            # apply mask over the data
            self.chanlabels = self.chanlabels[included_chans]
            self.rawdata = self.rawdata[included_chans]

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
        metadata['rawfilename'] = self.datafile
        metadata['patient'] = self.patient
        
        metadata['allchans'] = self.allchans
        
        # Set data from external text, connectivity, elec files
        if self.conn is not None:
            metadata['ez_region'] = self.conn.region_labels[self.ezinds]
            metadata['region_labels'] = self.conn.region_labels
            metadata['chanxyz'] = self.chanxyz
            metadata['contact_regs'] = self.contact_regs
        try:
            metadata['clipper_buffer_sec'] = self.buffer_sec
        except Exception as e:
            self.logger.info(e)
        try:
            metadata['chunklist'] = self.winlist
            metadata['secsperchunk'] = self.secsperchunk
        except BaseException:
            self.logger.info(
                'chunking not set for %s %s \n' %
                (self.patient, self.datafile))
            
        try:
            metadata['included_chans'] = self.included_chans
        except BaseException:
            self.logger.info(
                'included_chans not set for %s %s \n' %
                (self.patient, self.datafile))

        return metadata

    def _chunks(self, l, n):
        """
        Yield successive n-sized chunks from l.
        """
        for i in range(0, len(l), n):
            chunk = l[i:i + n]
            yield [chunk[0], chunk[-1]]

    def clipdata(self, ind=None):
        """
        Function to clip the data. It could either returns:
             a generator through the data, or
             just returns data at that index through the index

        See code below.

        """
        # if ind is None:
        #     # produce a generator that goes through the window list
        #     for win in self.winlist:
        #         data, times = self.raw[:,win[0]:win[-1]+1]
        #         yield data, times
        # else:
        # get info dict
        win = self.winlist[ind]
        data, times = self.raw[:, win[0]:win[1] + 1]
        return data, times

    def computechunks(self, secsperchunk=60):
        """
        Function to compute the chunks through the data by intervals of 60 seconds.

        This can be useful for sifting through the data one range at time.

        Note: The resolution for any long-term frequency analysis would be 1/60 Hz,
        which is very low, and can be assumed to be DC anyways when we bandpass filter.
        """
        self.secsperchunk = secsperchunk
        samplerate = self.samplerate
        numsignalsperwin = np.ceil(secsperchunk * samplerate).astype(int)

        numsamps = self.raw.n_times
        winlist = []

        # define a lambda function to subtract window
        def winlen(x): return x[1] - x[0]
        for win in self._chunks(np.arange(0, numsamps), numsignalsperwin):
            # ensure that each window length is at least numsignalsperwin
            if winlen(win) < numsignalsperwin - 1:
                winlist[-1][1] = win[1]
            else:
                winlist.append(win)
        self.winlist = winlist

    def clipdata_setwins(self, buffer_sec=60):
        onsetind = self.onset_ind
        offsetind = self.offset_ind 

        # get the onsets/offset necessary
        preonset = int(onsetind - buffer_sec*self.samplerate)
        newonsetind = int(buffer_sec*self.samplerate)
        if preonset < 0:
            preonset = 0
            newonsetind = onsetind

        numsamps = self.raw.n_times
        postoffset = int(offsetind + buffer_sec*self.samplerate)
        newoffsetind = int(offsetind + buffer_sec*self.samplerate)
        if postoffset > numsamps:
            postoffset = numsamps
            newoffsetind = offsetind

        data, times = self.raw[:, preonset:postoffset]
        self.onset_ind = newonsetind
        self.offset_ind = newoffsetind
        self.onset_sec = self.onset_ind / self.samplerate
        self.offset_sec = self.offset_ind / self.samplerate
        self.buffer_sec = buffer_sec

        return data, times

    def clipinterictal(self, samplesize_sec=60):
        samplesize = int(samplesize_sec * self.samplerate)
        numsamps = self.raw.n_times

        # use random clip of the interictal data
        # randomstart = np.random.choice(0, numsamps-samplesize, 1)
        # winclip = np.arange(randomstart, randomstart+samplesize).astype(int)
        # use clip at the beginning of the dataset
        winclip = np.arange(0, samplesize).astype(int)

        data, times = self.raw[:, 0:samplesize]
        return data, times

