import os
import json
import io
import zipfile
import numpy as np

from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger
from tvbsim.base.utils.data_structures_utils import NumpyEncoder
from tvbsim.io.utils import utils
from tvbsim.io.readers.read_surf import LoadSurface 
from tvbsim.io.readers.read_connectivity import LoadConn
from .contacts import Contacts

# to allow compatability between python2/3
try:
    to_unicode = unicode
except NameError:
    to_unicode = str


class BaseSubjectLoader(object):
    # generally optional data depending on how patient was analyzed
    # derived from MRI+CT
    chanxyzlabels = [] 
    chanxyz = []       
    # derived from MRI+CT+DWI
    contact_regs = []
    # derived from connectivity
    conn = None
    weights = np.array([])
    tract_lengths = np.array([])
    region_centres = np.array([])
    region_labels = []
    # surface object
    surf = None
    # ez hypothesis by clinicians
    ezinds = []

    # default atlas for loading in parcellation
    atlas = 'dk'

    ez_hyp_file = ''
    connfile = ''
    surfacefile = ''
    label_volume_file = ''

    def __init__(self, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
                        self.__class__.__name__,
                        self.config.out.FOLDER_LOGS)

    def _exists(self, filepath):
        return os.path.exists(filepath)

    def _get_channel_mask(self, labels, masks):
        '''
        Helper function to get the overlapping masks
        on the list of labels provided.

        Masks is a list of masks. For ex:
        _bad_channel_mask = np.array([ch in self.bad_channels for ch in self.chanlabels], dtype=bool)
        masks = [_bad_channel_mask, ...etc.]
        '''
        mask = np.ones(len(labels), dtype=bool)
        for _mask in masks:
            mask *= ~ _mask
        return mask

    def sync_gray_chans(self):
        # reject white matter contacts
        _gray_channels_mask = np.array([ch != -1 for ch in self.contact_regs], dtype=bool)
        mask = np.ones(len(self.contact_regs), dtype=bool)
        mask *= ~_gray_channels_mask
        return mask   

    def sync_xyz_and_raw(self, chanlabels, chanxyzlabels):
        '''             REJECT BAD CHANS LABELED BY CLINICIAN       '''
        # only deal with contacts both in raw data and with xyz coords
        _non_xyz_mask = np.array([ch not in chanxyzlabels for ch in chanlabels], dtype=bool)
        _non_data_mask = np.array([ch not in chanlabels for ch in chanxyzlabels], dtype=bool)

        rawdata_mask = np.ones(len(chanlabels), dtype=bool)
        xyzdata_mask = np.ones(len(chanxyzlabels), dtype=bool)

        rawdata_mask *= ~ _non_xyz_mask
        xyzdata_mask *= ~ _non_data_mask
        return rawdata_mask, xyzdata_mask

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
            self.logger.debug("\nAlready renamed gaininv.mat possibly!\n")

    def _init_files(self, setfiledir=True):
        '''
        Initialization function to be called
        '''
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
        if not self._exists(self.sensorsfile):
            self.sensorsfile = os.path.join(self.elecdir , 'seeg.xyz')
        
        # label volume file for where each contact is
        self.label_volume_file = os.path.join(self.dwidir, "label_in_T1.%s.nii.gz" % self.atlas)
        if not self._exists(self.label_volume_file):
            self.label_volume_file = os.path.join(self.dwidir, "label_in_T1.nii.gz")
        
        # connectivity file
        self.connfile = os.path.join(self.tvbdir, "connectivity.%s.zip" % self.atlas)
        if not self._exists(self.connfile):
            self.connfile = os.path.join(self.tvbdir, "connectivity.zip")

        # surface geometry file
        self.surfacefile = os.path.join(self.tvbdir, "surface_cort.%s.zip" % self.atlas)
        if not self._exists(self.connfile):
            self.surfacefile = os.path.join(self.tvbdir, "surface_cort.zip")

        # computed gain matrix file
        self.gainfile = os.path.join(self.elecdir, 'gain-in-square.txt')
        if not os.path.exists(self.sensorsfile):
            self.gainfile = os.path.join(self.elecdir, 'gain-in-square.%s.txt' % self.atlas)

        self.ez_hyp_file = os.path.join(self.tvbdir, 'ez_hypothesis.txt')
        if not os.path.exists(self.ez_hyp_file):
            self.ez_hyp_file = os.path.join(self.tvbdir, 'ez_hypothesis.dk.txt')

    def _mapcontacts_toregs(self):
        if not self._exists(self.label_volume_file):
            return

        self.contact_regs = np.array(
            utils.mapcontacts_toregs_v2(self.contacts, self.label_volume_file))
        self.logger.debug("\nMapped contacts to regions!\n")

    def _loadseegxyz(self):
        if not self._exists(self.sensorsfile):
            return

        seeg_pd = utils.loadseegxyz(self.sensorsfile)
        self.chanxyzlabels = np.array(seeg_pd.index.values)
        self.chanxyz = seeg_pd.as_matrix(columns=None)
        self.logger.debug("\nLoaded in seeg xyz coords!\n")

    def _loadcontacts(self):
        self.contacts = Contacts(self.seegfile)

    def _loadezhypothesis(self):
        if not self._exists(self.ez_hyp_file):
            return
        self.ez_hypothesis = np.genfromtxt(self.ez_hyp_file,
                                           dtype=int).astype(bool)
        self.ezinds = np.where(self.ez_hypothesis == 1)[0]

    def _loadconnectivity(self):
        if not self._exists(self.connfile):
            return

        self.conn = LoadConn().readconnectivity(self.connfile)
        self.weights = self.conn.weights
        self.region_labels = self.conn.region_labels
        self.region_centres = self.conn.centres
        self.tract_lengths = self.conn.tract_lengths

    def _loadsurface(self):
        if not self._exists(self.surfacefile):
            return
        self.surf = LoadSurface().loadsurfdata(self.tvbdir, use_subcort=False)

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
        except Exception as e:
            print(e)
            print("can't open metafile: {}".format(metafilename))
            with io.open(metafilename, encoding='utf-8', mode='r') as fp:
                json_str = fp.read() #json.loads(
            metadata = json.loads(json_str)

        self.metadata = metadata
        return self.metadata

    def _check_all_files(self):
        print("Checking tvb dir: ", self._exists(self.tvbdir))
        print("Checking seeg file: ", self._exists(self.seegfile))
        print("Checking label volume file: ", self._exists(self.label_volume_file))
        print("Checking connectivity file: ", self._exists(self.connfile))
        print("Checking ez hypothesis file: ", self._exists(self.ez_hyp_file))
        # print("Checking surface file: ", self._exists(self.surfacefile))

    def clipseizure(self, buffer_sec=60):
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

        data, _ = self.raw[:, preonset:postoffset]
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
        def _chunks(l, n):
            """
            Yield successive n-sized chunks from l.
            """
            for i in range(0, len(l), n):
                chunk = l[i:i + n]
                yield [chunk[0], chunk[-1]]

        self.secsperchunk = secsperchunk
        samplerate = self.samplerate
        numsignalsperwin = np.ceil(secsperchunk * samplerate).astype(int)

        numsamps = self.raw.n_times
        winlist = []

        # define a lambda function to subtract window
        def winlen(x): return x[1] - x[0]
        for win in _chunks(np.arange(0, numsamps), numsignalsperwin):
            # ensure that each window length is at least numsignalsperwin
            if winlen(win) < numsignalsperwin - 1 and winlist:
                winlist[-1][1] = win[1]
            else:
                winlist.append(win)
        self.winlist = winlist

        self.metadata['chunklist'] = self.winlist
        self.metadata['secsperchunk'] = self.secsperchunk
       
