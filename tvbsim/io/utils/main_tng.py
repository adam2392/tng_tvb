import numpy as np 
import pandas as pd 
from .loadpat import LoadPat
import warnings
import os

warnings.filterwarnings('ignore')

class LoadData(object):
    # @staticmethod
    def getdata(self, patdatadir, metafile):    
        patloader = LoadPat(patdatadir)
        if not metafile.endswith('.json'):
            metafile += '.json'
            
        metadata = patloader.loadmetadata(metafile)
        seeg_pd = patloader.loadseegxyz()
        contact_regs = patloader.mapcontacts_toregs()

        # get chan xyz from seeg df
        chanxyz_labs = seeg_pd.index.values
        chanxyz = seeg_pd.as_matrix(columns=None)

        # load in the useful metadata
        rawfile = metadata['filename']
        onsetsec = metadata['onset']
        offsetsec = metadata['termination']
        badchans = metadata['bad_channels']
        try:
            nonchans = metadata['non_seeg_channels']
        except:
            nonchans = []
        badchans = badchans + nonchans 
        sztype = metadata['type']

        # create the mne object
        rawdata_mne = patloader.loadrawdata(rawfile, reference='monopolar')

        # apply mne notch filter
        samplerate = rawdata_mne.info['sfreq']
        freqs = np.arange(50,251,50) 
        # np.arange(60,241,60)) # if USA
        freqs = np.delete(freqs, np.where(freqs>samplerate//2)[0])
        rawdata_mne.notch_filter(freqs=freqs) # if Europe

        # extract the actual SEEG time series
        rawdata = rawdata_mne.get_data()
        print(rawdata.shape)
        
        chanlabels = np.array(rawdata_mne.ch_names)

        # map badchans, chanlabels to lower case
        badchans = np.array([lab.lower() for lab in badchans])
        chanxyz_labs = np.array([lab.lower() for lab in chanxyz_labs])
        chanlabels = np.array([lab.lower() for lab in chanlabels])

        # extract necessary metadata
        goodchans_inds = [idx for idx,chan in enumerate(chanlabels) if chan not in badchans if chan in chanxyz_labs]
        
        print(rawdata.shape)
        print(contact_regs.shape)
        print(chanlabels.shape)
        print(chanxyz.shape)
        print(len(goodchans_inds))

        # only grab the good channels specified
        goodchans = chanlabels[goodchans_inds]
        rawdata = rawdata[goodchans_inds,:]

        # now sift through our contacts with xyz coords and region_mappings
        reggoodchans_inds = [idx for idx,chan in enumerate(chanxyz_labs) if chan in goodchans]
        contact_regs = contact_regs[reggoodchans_inds]
        chanxyz = chanxyz[reggoodchans_inds,:]        

        print('getting gray matter contacts now')
        print(contact_regs.shape)
        print(goodchans.shape)
        print(rawdata.shape)
        print(chanxyz.shape)

        # reject white matter contacts
        graychans_inds = np.where(np.asarray(contact_regs) != -1)[0]

        print(len(graychans_inds))
        print(graychans_inds)

        contact_regs = contact_regs[graychans_inds]
        goodchans = goodchans[graychans_inds]
        rawdata = rawdata[graychans_inds,:]
        chanxyz = chanxyz[graychans_inds,:]

        self.offsetind = np.multiply(offsetsec, samplerate)
        self.onsetind = np.multiply(onsetsec, samplerate)

        self.chanlabels = goodchans
        self.rawdata = rawdata
        self.contact_regs = contact_regs
        self.samplerate = samplerate
        self.rawfile = rawfile
        self.onsetsec = onsetsec
        self.offsetsec = offsetsec
        self.chanxyz = chanxyz
        self.sztype = sztype