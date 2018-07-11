import os
import json
import io
import zipfile
import numpy as np
import pandas as pd

from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger
from tvbsim.base.utils.data_structures_utils import NumpyEncoder

from tvbsim.io.loaders.base.baseloaders import MetaLoaders
from tvbsim.io.config import dataconfig

# to allow compatability between python2/3
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

class StructuralDataLoader(object):
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
    regezinds = []

    # default atlas for loading in parcellation
    atlas = dataconfig.atlas

    ez_hyp_file = ''
    connfile = ''
    surfacefile = ''
    label_volume_file = ''

    def __init__(self, root_dir, atlas=None, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
                        self.__class__.__name__,
                        self.config.out.FOLDER_LOGS)

        # initialize our loader class to load all sorts of raw data
        self.rawloader = MetaLoaders(self.config, self.logger)

        if atlas is not None:
            self.atlas = atlas

        self.root_dir = root_dir
        self._init_files()

        # load all files
        self.load_raw_meta()

    def load_raw_meta(self):
        # load in connectivity
        if os.path.exists(self.connfile):
            self.conn = self.rawloader._loadconnectivity(self.connfile)
        # load in ez_hypothesis
        if os.path.exists(self.ez_hyp_file):
            self.regezinds = self.rawloader._loadezhypothesis(self.ez_hyp_file)
        # load in surface
        if os.path.exists(self.surfacefile) and os.path.exists(self.regionmapfile): 
            print("Loading: ", self.surfacefile, self.regionmapfile)
            self.surface = self.rawloader._loadsurface(self.surfacefile, self.regionmapfile)
        # load in contacts
        # self.rawloader._loadcontacts()
        if os.path.exists(self.sensorsfile):
            self.chanxyz, self.chanxyzlabels = self.rawloader._loadseegxyz(self.sensorsfile)
        if os.path.exists(self.gainfile):
            self.gainmat = np.array(self.rawloader._loadgainmat(self.gainfile))
        # map contacts to regions using DWI and T1 Parcellation
        if os.path.exists(self.label_volume_file) and os.path.exists(self.sensorsfile): 
            self.contact_regs = np.array(self.rawloader._mapcontacts_toregs(self.label_volume_file, self.sensorsfile))

        # preprocess channels - lowercase, remove 'POL'
        self.chanxyzlabels = np.array(self.scrubchannels(self.chanxyzlabels))
        
        ''' create masks needed to "synchronize" data '''
        # mask over contact regions with gray matter
        if len(self.contact_regs) > 0:
            self.whitemattermask = self.create_data_masks(self.chanxyzlabels, self.contact_regs)
            metamask = [idx for idx, ch in enumerate(self.chanxyzlabels) \
                        if ch not in self.whitemattermask]
            self.chanxyzlabels = self.chanxyzlabels[metamask]
            self.chanxyz = self.chanxyz[metamask,:]
            self.contact_regs = self.contact_regs[metamask]

    def __renamefiles(self):
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
        # lambda dir, file: 
        if setfiledir:
            self.seegdir = os.path.join(self.root_dir, 'seeg', 'fif')
            if not os.path.exists(self.seegdir):
                self.seegdir = os.path.join(self.root_dir, 'seeg', 'edf')
            self.elecdir = os.path.join(self.root_dir, 'elec')
            self.dwidir = os.path.join(self.root_dir, 'dwi')
            self.tvbdir = os.path.join(self.root_dir, 'tvb')

        # assumes elec/tvb/dwi/seeg dirs are set
        self.__renamefiles()

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
        self.surfacefile = os.path.join(self.tvbdir, "surface_cort.zip")
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

    def scrubchannels(self, labels):
        labels = np.array([ch.lower() for ch in labels])
        labels = [str(x).replace('pol', '').replace(' ', '')
                           for x in labels]
        return labels

    def sync_xyz_and_raw(self, chanlabels, chanxyzlabels):
        '''             REJECT BAD CHANS LABELED BY CLINICIAN       '''
        # create mask from bad/noneeg channels
        self.badchannelsmask = self.bad_channels
        self.noneegchannelsmask = self.non_eeg_channels

        # create mask from raw recording data and structural data
        if len(self.chanxyzlabels) > 0:
            self.rawdatamask = np.array([ch for ch in self.chanlabels if ch not in self.chanxyzlabels])
        else:
            self.rawdatamask = np.array([])
        if len(self.chanlabels) > 0:
            self.xyzdatamask = np.array([ch for ch in self.chanxyzlabels if ch not in self.chanlabels])
        else:
            self.xyzdatamask = np.array([])

    def create_data_masks(self, chanxyzlabels, contact_regs):
        # create mask from raw recording data and structural data
        # reject white matter contacts
        # find channels that are not part of gray matter
        assert len(chanxyzlabels) == len(contact_regs)
        assert np.min(contact_regs) == -1 # to make sure that our minimum contact is -1 == white matter

        _white_channels_mask = np.array([idx for idx, regid in enumerate(contact_regs) if regid == -1])
        return chanxyzlabels[_white_channels_mask]