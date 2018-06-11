import numpy as np
import os
import pandas as pd
import mne
from datetime import date 
'''
Takes in a pair file of .ades and .dat and extracts the channel names and the corresponding SEEG time series

places them into four different files

- raw numpy
- headers csv
- annotations csv
- channels csv

Which follows format that we place data from .edf files. Most data is empty since .ades does not get alot of these
data points.
'''

class ExtractAdes(object):
    def __init__(self, filepath):
        self.filepath = filepath

        direct, fname = os.split(filepath)
        self.read_ades(filepath)

    def read_ades(self, fname):
        dat_fname = fname.split('.ades')[0] + '.dat'
        srate = None
        nsamp = None
        sensors = []
        with open(fname, 'r') as fd:
            for line in fd.readlines():
                if line.startswith('#'):
                    continue
                # parts = line.strip().split(' ')
                # lhs = parts[0]
                # rhs = parts[2]

                try:
                    lhs, _, rhs = line.strip().split(' ')
                except ValueError:
                    lhs = line.strip().split(' ')

                if lhs == 'samplingRate':
                    srate = float(rhs)
                elif lhs == 'numberOfSamples':
                    nsamp = float(rhs)
                elif lhs in ('date', 'time'):
                    pass
                else:
                    if isinstance(lhs, list):
                        lhs = lhs[0]
                    sensors.append(lhs)
        assert srate and nsamp
        dat = np.fromfile(
            dat_fname, dtype=np.float32).reshape(
            (-1, len(sensors))).T
        return srate, sensors, dat, nsamp

    def extract_data_edf(self, filepath):
        self.raw = mne.io.read_raw_edf(filepath, preload=True,
                                          verbose=False)
        # get events
        if datafilepath.endswith('.edf'):
            events = self.raw.find_edf_events()
            self.events = np.array(events)
            self.__setevents(self.events)

    def _loadinfodata(self):
        # set samplerate
        self.samplerate = self.raw.info['sfreq']
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

        # iterate through the events and assign onset/offset times if avail.
        for idx, name in enumerate(eventnames):
            name = name.lower().split(' ')
            if 'onset' in name \
                    or 'crise' in name \
                    or 'cgtc' in name \
                    or 'sz' in name or 'absence' in name:
                if not onsetset:
                    onset_secs = eventonsets[idx]
                    onsetset = True
            if 'offset' in name \
                    or 'fin' in name \
                    or 'end' in name:
                if not offsetset:
                    offset_secs = eventonsets[idx]
                    offsetset = True

        # set onset/offset times and markers
        try:
            self.onset_sec = onset_secs
            self.onset_ind = np.ceil(onset_secs * self.samplerate)
        except TypeError:
            self.logger.info("no onset time!")
            self.onset_sec = None
            self.onset_ind = None
        try:
            self.offset_sec = offset_secs
            self.offset_ind = np.ceil(offset_secs * self.samplerate)
        except TypeError:
            self.logger.info("no offset time!")
            self.offset_sec = None
            self.offset_ind = None

