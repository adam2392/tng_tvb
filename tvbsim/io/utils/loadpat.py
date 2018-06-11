import sys
import os
import numpy as np
import pandas as pd
import nibabel as nib
import mne
import json

from .elecs import Contacts
from . import nifti
from .seegrecording import SeegRecording

'''
Example Run:

patient = 'id001_ac'
filepath = ''
filename = ''

patloader = LoadPat(patient, filepath)
metadata = patloader.loadmetadata(filename)
seeg_pd = patloader.loadseegxyz(seegfile)
contact_regs = patloader.mapcontacts_toregs(seegfile, label_volume_file)
rawdata = patloader.loadrawdata(filename)

'''


class LoadPat(object):
    '''
    A class describing a dataset that is within our framework. This object will help set up the
    data for computation and any meta data necessary to link the computations together.

    This assumes that data is stored in a specific format!
    datadir/
        <patient1>
        <patient2>
            - <p>_rawnpy.npy
            - <p>_annotations.csv
            - <p>_chans.csv
            - <p>_headers.csv

    This mainly is the format that is the output of the dataconversion.py file from EDF files.

    Will need to reformat for other type of EEG files.
    '''

    def __init__(self, filepath):
        self.filepath = filepath

    def loadmetadata(self, metafile):
        metafilename = os.path.join(self.filepath, 'seeg/fif', metafile)
        metafile = open(metafilename)
        metajson = metafile.read()

        self.metadata = json.loads(metajson)
        return self.metadata

    def loadseegxyz(self):
        '''
        This is just a wrapper function to retrieve the seeg coordinate data in a pd dataframe
        '''
        sensorsfile = os.path.join(self.filepath, 'elec/seeg.xyz')
        newsensorsfile = os.path.join(self.filepath, 'elec/seeg.txt')
        try:
            # copyfile(sensorsfile, newsensorsfile)
            os.rename(sensorsfile, newsensorsfile)
        except BaseException:
            print("Already renamed seeg.xyz possibly!")

        seegfile = os.path.join(self.filepath, 'elec/seeg.txt')
        seeg_pd = pd.read_csv(
            seegfile,
            names=[
                'x',
                'y',
                'z'],
            delim_whitespace=True)

        self.chanlabels = seeg_pd.index.values
        self.chanxyz = seeg_pd.as_matrix(columns=None)
        return seeg_pd

    def mapcontacts_toregs(self):
        sensorsfile = os.path.join(self.filepath, 'elec/seeg.xyz')
        newsensorsfile = os.path.join(self.filepath, 'elec/seeg.txt')
        try:
            # copyfile(sensorsfile, newsensorsfile)
            os.rename(sensorsfile, newsensorsfile)
        except BaseException:
            print("Already renamed seeg.xyz possibly!")

        contacts_file = os.path.join(self.filepath, 'elec/seeg.txt')
        label_volume_file = os.path.join(
            self.filepath, 'dwi/label_in_T1.nii.gz')

        contacts = Contacts(contacts_file)
        label_vol = nib.load(label_volume_file)

        contact_regs = []
        for contact in contacts.names:
            coords = contacts.get_coords(contact)
            region_ind = nifti.point_to_brain_region(
                coords, label_vol, tol=3.0) - 1   # Minus one to account for the shift
            contact_regs.append(region_ind)

        contact_regs = np.array(contact_regs)
        return contact_regs

    def loadrawdata(self, rawfile, reference='monopolar'):
        rawfilename = os.path.join(self.filepath, 'seeg/fif', rawfile)
        rawdata = mne.io.read_raw_fif(rawfilename)

        rawdata.load_data()
        if reference == 'avg':
            rawdata, ref_data = mne.set_eeg_reference(
                rawdata, ref_channels='average')
        elif reference == 'bipolar':
            print('NOT IMPLEMENTED YET')
            # rawdata = mne.set_bipolar_reference(rawdata)
        elif reference == 'monopolar':
            rawdata, ref_data = mne.set_eeg_reference(rawdata, ref_channels=[])

        return rawdata

    def _clipdata(self, rawdata):
        '''
        Used for clipping data at +/- 60 seconds of their onset/offset times

        To reduce the amount of data that is needed to analyze
        '''
        if self.onset_time:
            firstind = int(
                self.onset_sec *
                self.samplerate -
                60 *
                self.samplerate)
        if self.offset_time:
            endind = int(
                self.offset_sec *
                self.samplerate +
                60 *
                self.samplerate)
        else:
            endind = firstind + 60 * 1000

        # firstind and endind needs to be within the rawdata bounds
        self.firstind = firstind
        self.endind = endind

        # clip the rawdata
        rawdata = rawdata[:, firstind:endind]
        clippedinds = firstind
        clippedsec = firstind / self.samplerate
        clippedms = firstind / self.samplerate * 1000

        # clip onsettime, offsettime
        self.onset_time = 60 * 1000
        self.offset_time = rawdata.shape[1] - 60 * 1000
        self.onset_sec = self.onset_sec - 60
        self.offset_sec = self.offset_sec - clippedsec

        return rawdata

    def _clippeddatagen(self, rawdata, winsec=60):
        '''
        A data generator that goes through the raw data
        and returns clipped windows that have a predetermined
        number of seconds.

        To Do:
        - Test that beginind, endind always are continuous
        and there is no data cutoff.
        '''

        # clip data by the number of seconds in winsec
        winsize = winsec * self.samplerate
        numchans, numsamps = rawdata.shape

        ind = 0
        while True:
            beginind = ind * winsize
            endind = (ind + 1) * winsize

            # check if end of data will be reached
            if endind >= numsamps:
                yield rawdata[:, beginind:]
            else:
                yield rawdata[:, beginind:endind]
