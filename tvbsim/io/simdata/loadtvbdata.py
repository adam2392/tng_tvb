import os
import json
import io
import zipfile
import numpy as np
import pandas as pd

from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger
from tvbsim.base.utils.data_structures_utils import NumpyEncoder
from tvbsim.io.utils import utils
from tvbsim.io.readers.read_surf import LoadSurface 
from tvbsim.io.readers.read_connectivity import LoadConn
from tvbsim.io.utils.elecs import Contacts

# to allow compatability between python2/3
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

class SynchronizeData(object):
    
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

    def sync_gray_chans(self, contact_regs):
        # reject white matter contacts
        _gray_channels_mask = np.array([ch != -1 for ch in contact_regs], dtype=bool)
        mask = np.ones(len(contact_regs), dtype=bool)
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
    ezinds = []

    # default atlas for loading in parcellation
    atlas = 'dk'

    ez_hyp_file = ''
    connfile = ''
    surfacefile = ''
    label_volume_file = ''

    def __init__(self, root_dir, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
                        self.__class__.__name__,
                        self.config.out.FOLDER_LOGS)

        self.root_dir = root_dir
        self._init_files()

        # load in connectivity
        self._loadconnectivity()
        # load in ez_hypothesis
        self._loadezhypothesis()
        # load in surface
        self._loadsurface()
        # load in gain matrix
        self._loadgainmat()
        # load in contacts
        if os.path.exists(self.sensorsfile):
            self._loadcontacts()
            self._loadseegxyz()
        # map contacts to regions using DWI and T1 Parcellation
        if os.path.exists(self.label_volume_file):
            self._mapcontacts_toregs()

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

    def _loadgainmat(self):
        if not os.path.exists(self.gainfile):
            self.logger.error("Can't from {} because doesn't exist".format(self.gainfile))
            return
        # function to get model in its equilibrium value
        gain_pd = pd.read_csv(self.gainfile, header=None, delim_whitespace=True)
        self.gainmat = gain_pd.as_matrix()

    def _mapcontacts_toregs(self):
        if not os.path.exists(self.label_volume_file):
            self.logger.error("Can't from {} because doesn't exist".format(self.label_volume_file))
            return

        self.contact_regs = np.array(
            utils.mapcontacts_toregs_v2(self.contacts, self.label_volume_file))
        self.logger.debug("\nMapped contacts to regions!\n")

    def _loadseegxyz(self):
        if not os.path.exists(self.sensorsfile):
            self.logger.error("Can't from {} because doesn't exist".format(self.sensorsfile))
            return

        seeg_pd = utils.loadseegxyz(self.sensorsfile)
        self.chanxyzlabels = np.array(seeg_pd.index.values)
        self.chanxyz = seeg_pd.as_matrix(columns=None)
        self.logger.debug("\nLoaded in seeg xyz coords!\n")

    def _loadcontacts(self):
        self.contacts = Contacts(self.sensorsfile)

    def _loadezhypothesis(self):
        if not os.path.exists(self.ez_hyp_file):
            self.logger.error("Can't from {} because doesn't exist".format(self.ez_hyp_file))
            return
        self.ez_hypothesis = np.genfromtxt(self.ez_hyp_file,
                                           dtype=int).astype(bool)
        self.ezinds = np.where(self.ez_hypothesis == 1)[0]

    def _loadconnectivity(self):
        if not os.path.exists(self.connfile):
            self.logger.error("Can't from {} because doesn't exist".format(self.connfile))
            return

        self.conn = LoadConn().readconnectivity(self.connfile)
        self.weights = self.conn.weights
        self.region_labels = self.conn.region_labels
        self.region_centres = self.conn.centres
        self.tract_lengths = self.conn.tract_lengths

    def _loadsurface(self):
        if not os.path.exists(self.surfacefile) or not os.path.exists(self.regionmapfile):
            self.logger.error("Can't from {}, {} because doesn't exist".format(self.surfacefile, self.regionmapfile))
            return
        self.surf = LoadSurface().loadsurfdata(self.surfacefile, self.regionmapfile, use_subcort=False)

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
