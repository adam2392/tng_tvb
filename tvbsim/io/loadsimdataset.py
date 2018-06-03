import os
import numpy as np 
import pandas as pd 
import mne
import json

from tvbsim.io.utils import seegrecording
from tvbsim.io.base import BaseLoader
from tvbsim.base.preprocess.util.noise import LineNoise
from tvbsim.base.utils.data_structures_utils import NumpyEncoder
from datetime import date

class LoadSimDataset(BaseLoader):
    patient = None

    def __init__(self, rawdatadir, patient=None, config=None):
        super(LoadSimDataset, self).__init__(rawdatadir=rawdatadir, patient=patient, config=config)
        self.patient = patient 
       
    def load_data(self, simdata):
        self.data = simdata 

    def addlinenoise(self, rawdata):
        self.linefreq = 60
        bandwidth = 4
        numharmonics = 4
        # initialize line noise object
        noise = LineNoise(self.linefreq, 
                            bandwidth, 
                            numharmonics, 
                            self.samplerate)

        numsamps = self.rawdata.shape[1]

        self.logger.debug("Adding line noise at {} hz with {} harmonics".format(self.linefreq, numharmonics))
        # create a copy and add noise to it
        test = self.rawdata.copy()
        for i in range(rawdata.shape[0]):
            test[i, :] += noise.generate(numsamps)
        self.rawdata = test

    def filter_data(self):
        # the bandpass range to pass initial filtering
        freqrange =  [0.5, self.samplerate//2 - 1]
        # the notch filter to apply at line freqs
        linefreq = int(self.linefreq)           # LINE NOISE OF HZ
        assert linefreq == 50 or linefreq == 60
        self.logger.debug("Line freq is: %s" % linefreq)
        # initialize the line freq and its harmonics
        freqs = np.arange(linefreq,251,linefreq)
        freqs = np.delete(freqs, np.where(freqs>self.samplerate//2)[0])

        self.rawdata = mne.filter.filter_data(self.rawdata,
                                        sfreq=self.samplerate,
                                        l_freq=freqrange[0],
                                        h_freq=freqrange[1],
                                        # pad='reflect',
                                        verbose=False)
        self.rawdata = mne.filter.notch_filter(self.rawdata,
                                        Fs=self.samplerate,
                                        freqs=freqs,
                                        verbose=False)

    def savejsondata(self, metadata, metafilename):
        # save the timepoints, included channels used, parameters
        dumped = json.dumps(metadata, cls=NumpyEncoder)
        with open(metafilename, 'w') as f:
            json.dump(dumped, f)
        self.logger.info('Saved metadata as json!')

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

        # needs to be gotten from sync_data()
        # metadata['goodchans'] = dataloader.goodchans
        # metadata['graychans'] = dataloader.graychans_inds

        return metadata