import os
import numpy as np
import pandas as pd
import json
import zipfile
from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger
from tvbsim.base.utils.data_structures_utils import NumpyEncoder
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

from tvbsim.io.loaders.base.baseloaders import MetaLoaders
from tvbsim.io.config import dataconfig

class BaseMetaLoader(object):
    # generally optional data depending on how patient was analyzed
    # derived from MRI+CT
    chanxyzlabels = np.array([])
    chanxyz = np.array([])
    # derived from MRI+CT+DWI
    contact_regs = np.array([])
    # derived from connectivity
    conn = None
    weights = np.array([])
    tract_lengths = np.array([])
    region_centres = np.array([])
    region_labels = np.array([])
    # surface object
    surf = None
    # ez hypothesis by clinicians
    regezinds = np.array([])

    # default atlas for loading in parcellation
    atlas = dataconfig.atlas

    ez_hyp_file = ''
    connfile = ''
    surfacefile = ''
    label_volume_file = ''
    def __init__(self, root_dir, config=None):
        self.root_dir = root_dir
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

        self.logger.debug("Setting atlas via dataconfig as {}".format(self.atlas))
    
        # initialize our loader class to load all sorts of raw data
        self.rawloader = MetaLoaders(self.config, self.logger)

        # run initialization of files
        self._init_files()

        # print some debugging messages about files
        self._check_all_files()

        # load all files
        self.load_raw_meta()

    def load_raw_meta(self):
        # load in connectivity
        self.conn = self.rawloader._loadconnectivity(self.connfile)
        # load in ez_hypothesis
        self.regezinds = self.rawloader._loadezhypothesis(self.ez_hyp_file)
        # load in surface
        # self.surface = self.rawloader._loadsurface(self.surfacefile, self.regionmapfile)
        
        # load in contacts
        # self.rawloader._loadcontacts()
        if os.path.exists(self.sensorsfile):
            self.logger.error("Can't from {} because doesn't exist".format(self.sensorsfile))
            self.chanxyz, self.chanxyzlabels = self.rawloader._loadseegxyz(self.sensorsfile)
        if os.path.exists(self.gainfile):
            self.logger.error("Can't from {} because doesn't exist".format(self.gainfile))
            self.gainmat = np.array(self.rawloader._loadgainmat(self.gainfile))
        # map contacts to regions using DWI and T1 Parcellation
        if os.path.exists(self.sensorsfile) and os.path.exists(self.label_volume_file):
            self.logger.error("Can't from {} because doesn't exist".format(self.label_volume_file))
            self.contact_regs = np.array(self.rawloader._mapcontacts_toregs(self.label_volume_file, self.sensorsfile))

        # preprocess channels - lowercase, remove 'POL'
        self.chanxyzlabels = np.array(self.scrubchannels(self.chanxyzlabels))
        
        ''' create masks needed to "synchronize" data '''
        # mask over contact regions with gray matter
        if len(self.contact_regs) > 0:
            self.whitemattermask = self.sync_gray_chans(self.contact_regs)
            # masks.append(whitemattermask)

    def sync_gray_chans(self, contact_regs):
        # reject white matter contacts
        # find channels that are not part of gray matter
        assert len(self.chanxyzlabels) == len(contact_regs)
        assert np.min(contact_regs) == -1 # to make sure that our minimum contact is -1 == white matter

        _white_channels_mask = np.array([idx for idx, regid in enumerate(contact_regs) if regid == -1])
        return self.chanxyzlabels[_white_channels_mask]
        # return _white_channels_mask 

    def _init_files(self, setfiledir=True):
        '''
        Initialization function to be called
        '''
        # lambda dir, file: 
        if setfiledir:
            self.seegdir = os.path.join(self.root_dir, 'seeg', 'fif')
            if not os.path.exists(self.seegdir):
                self.seegdir = os.path.join(self.root_dir, 'seeg', 'edf')
            self.elecdir = os.path.join(self.root_dir, 'elec')
            self.dwidir = os.path.join(self.root_dir, 'dwi')
            self.tvbdir = os.path.join(self.root_dir, 'tvb')
        # assumes elec/tvb/dwi/seeg dirs are set
        self._renamefiles()

        # sensors file with xyz coords
        self.sensorsfile = os.path.join(self.elecdir , 'seeg.txt')
        if not os.path.exists(self.sensorsfile):
            self.sensorsfile = os.path.join(self.elecdir , 'seeg.xyz')
        
        # label volume file for where each contact is
        self.label_volume_file = os.path.join(self.dwidir, "label_in_T1.%s.nii.gz" % self.atlas)
        if not os.path.exists(self.label_volume_file):
            self.label_volume_file = os.path.join(self.dwidir, "label_in_T1.nii.gz")
        
        # connectivity file
        self.connfile = os.path.join(self.tvbdir, "connectivity.%s.zip" % self.atlas)
        if not os.path.exists(self.connfile):
            self.connfile = os.path.join(self.tvbdir, "connectivity.zip")

        # surface geometry file
        self.surfacefile = os.path.join(self.tvbdir, "surface_cort.%s.zip" % self.atlas)
        if not os.path.exists(self.surfacefile):
            self.surfacefile = os.path.join(self.tvbdir, "surface_cort.zip")

        self.regionmapfile = os.path.join(self.tvbdir, "region_mapping_cort.%s.txt" % self.atlas)
        if not os.path.exists(self.regionmapfile):
            self.regionmapfile = os.path.join(self.tvbdir, "region_mapping_cort.txt")

        # computed gain matrix file
        self.gainfile = os.path.join(self.elecdir, 'gain_inv-square.txt')
        if not os.path.exists(self.gainfile):
            self.gainfile = os.path.join(self.elecdir, 'gain_inv-square.%s.txt' % self.atlas)

        self.ez_hyp_file = os.path.join(self.tvbdir, 'ez_hypothesis.txt')
        if not os.path.exists(self.ez_hyp_file):
            self.ez_hyp_file = os.path.join(self.tvbdir, 'ez_hypothesis.dk.txt')

    def _renamefiles(self):
        sensorsfile = os.path.join(self.elecdir, 'seeg.xyz')
        newsensorsfile = os.path.join(self.elecdir, 'seeg.txt')
        try:
            # copyfile(sensorsfile, newsensorsfile)
            os.rename(sensorsfile, newsensorsfile)
        except BaseException:
            self.logger.debug("\nAlready renamed seeg.xyz possibly!\n")
        gainfile = os.path.join(self.elecdir, 'gain_inv-square.mat')
        newgainfile = os.path.join(self.elecdir, 'gain_inv-square.txt')
        try:
            os.rename(gainfile, newgainfile)
        except BaseException:
            self.logger.debug("\nAlready renamed gain.mat possibly!\n")
        self.sensorsfile = newsensorsfile

    def _check_all_files(self):
        self.logger.debug("Checking tvb dir: ", os.path.exists(self.tvbdir))
        self.logger.debug("Checking seeg file: ", os.path.exists(self.sensorsfile))
        self.logger.debug("Checking label volume file: ", os.path.exists(self.label_volume_file))
        self.logger.debug("Checking connectivity file: ", os.path.exists(self.connfile))
        self.logger.debug("Checking ez hypothesis file: ", os.path.exists(self.ez_hyp_file))
        self.logger.debug("Checking surface file: ", os.path.exists(self.surfacefile))

    def scrubchannels(self, labels):
        labels = np.array([ch.lower() for ch in labels])
        labels = [str(x).replace('pol', '').replace(' ', '')
                           for x in labels]
        return labels

    def _writejsonfile(self, metadata, metafilename):
        with io.open(metafilename, 'w', encoding='utf8') as outfile:
            str_ = json.dumps(metadata,
                              indent=4, sort_keys=True, cls=NumpyEncoder,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

    def _loadjsonfile(self, metafilename):
        if not metafilename.endswith('.json'):
            metafilename += '.json'

        try:
            with open(metafilename, mode='r', encoding='utf8') as f:
                metadata = json.load(f)
            metadata = json.loads(metadata)
        except:
            with io.open(metafilename, encoding='utf-8', mode='r') as fp:
                json_str = fp.read() #json.loads(
            metadata = json.loads(json_str)

        # self.metadata = metadata
        return metadata

    def clipseizure(self, buffer_sec=30):
        onsetind = self.onset_ind
        offsetind = self.offset_ind 
        buffersamps = buffer_sec * self.samplerate
        numsamps = self.raw.n_times

        # get the data clips
        preclip = int(onsetind - buffersamps)
        postclip = int(offsetind + buffersamps)

        # reassign the onsets/offset accordingly
        newonsetind = int(buffersamps)
        newoffsetind = int(buffersamps + (offsetind - onsetind))

        if preclip < 0:
            preclip = 0
            newonsetind = onsetind
        if postclip > numsamps:
            postclip = numsamps

        data, _ = self.raw[:, preclip:postclip]

        print("Original raw data was: {}".format(self.raw.n_times))
        print("Original was: {} {}".format(onsetind, offsetind))
        print("Onset ind and offsetind are: {} {}".format(newonsetind, newoffsetind))
        print("Clipping from {}:{}".format(preclip, postclip))

        self.onset_ind = newonsetind
        self.offset_ind = newoffsetind
        self.onset_sec = self.onset_ind / self.samplerate
        self.offset_sec = self.offset_ind / self.samplerate
        self.buffer_sec = buffer_sec
        self.metadata['clipper_buffer_sec'] = self.buffer_sec
        return data

    def clipinterictal(self, samplesize_sec=60):
        samplesize = int(samplesize_sec * self.samplerate)

        # just get the first 60 seconds of the interictal clip
        winclip = np.arange(0, samplesize).astype(int)
        data, _ = self.raw[:, 0:samplesize]
        return data

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
            if winlen(win) < numsignalsperwin - 1 and winlist:
                winlist[-1][1] = win[1]
            else:
                winlist.append(win)
        self.winlist = winlist

    def _chunks(self, l, n):
        """
        Yield successive n-sized chunks from l.
        """
        for i in range(0, len(l), n):
            chunk = l[i:i + n]
            yield [chunk[0], chunk[-1]]

    def clipdata_chunks(self, ind=None):
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

        win = self.winlist[ind]
        data, times = self.raw[:, win[0]:win[1] + 1]
        return data, times

    @staticmethod
    def load_good_chans_inds(self, chanlabels, contact_regs=[], chanxyzlabels=[],
                            bad_channels=[], non_eeg_channels=[]):
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
        if len(chanxyzlabels) > 0:
            # sync up xyz and rawdata
            rawdata_mask, _ = self.sync_xyz_and_raw(chanlabels, chanxyzlabels)
            # reject white matter contacts
            if len(contact_regs) > 0:
                white_matter_chans = np.array([ch for idx, ch in enumerate(chanxyzlabels) if contact_regs[idx] == -1])
                _gray_channels_mask = np.array([ch in white_matter_chans for ch in chanlabels], dtype=bool)

        # reject bad channels and non-seeg contacts
        _bad_channel_mask = np.array([ch in bad_channels for ch in chanlabels], dtype=bool)
        _non_seeg_channels_mask = np.array([ch in non_eeg_channels for ch in chanlabels], dtype=bool)
        rawdata_mask *= ~_bad_channel_mask
        rawdata_mask *= ~_non_seeg_channels_mask
        rawdata_mask *= ~_gray_channels_mask
        return rawdata_mask