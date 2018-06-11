import mne
import numpy as np 
from datetime import date 
import warnings

warnings.filterwarnings('ignore')
class ExtractEDF(object):
    onset_sec = None
    onset_ind = None
    offset_sec = None
    offset_ind = None

    def __init__(self, filepath):
        self.extract_data_edf(filepath)

    def extract_data_edf(self, filepath):
        self.raw = mne.io.read_raw_edf(filepath, preload=True,
                                          verbose=False)
        self._loadinfodata()
        # get events
        if filepath.endswith('.edf'):
            events = self.raw.find_edf_events()
            self.events = np.array(events)
            if len(self.events) > 0:
                self.__setevents(self.events)

    def _loadinfodata(self):
        # set samplerate
        self.samplerate = float(self.raw.info['sfreq'])
        # set channel names
        self.chanlabels = self.raw.info['ch_names']
        self._processchanlabels()

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

        onsetmarks = ['onset', 'crise', 'cgtc', 'sz', 'absence']
        offsetmarks = ['offset', 'fin', 'end']
        # iterate through the events and assign onset/offset times if avail.
        for idx, name in enumerate(eventnames):
            name = name.lower().split(' ')
            if any(x in name for x in onsetmarks):
                if not onsetset:
                    onset_secs = float(eventonsets[idx])
                    onsetset = True
            if any(x in name for x in offsetmarks):
                if not offsetset:
                    offset_secs = float(eventonsets[idx])
                    offsetset = True

        # set onset/offset times and markers
        try:
            self.onset_sec = onset_secs
            self.onset_ind = np.ceil(np.multiply(onset_secs, self.samplerate))
        except TypeError as e:
            print(e)
            print("no onset time!")
            self.onset_sec = None
            self.onset_ind = None
        try:
            self.offset_sec = offset_secs
            self.offset_ind = np.ceil(np.multiply(offset_secs, self.samplerate))
        except TypeError as e:
            print(e)
            print("no offset time!")
            self.offset_sec = None
            self.offset_ind = None