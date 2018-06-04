import os
import numpy as np 
import pandas as pd 
import json
from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger
from tvbsim.io.utils import utils, seegrecording
from tvbsim.io.readers.read_connectivity import LoadConn
from tvbsim.io.readers.read_surf import LoadSurface

class BaseLoader(object):
    gainfile = None
    sensorsfile = None
    connfile = None
    label_volume_file = None
    ez_hyp_file = None

    elec_dir = None
    seegdir = None
    dwidir = None
    tvbdir = None

    conn = None
    ezinds = None
    chanxyz = None
    chanxyz_labels = None
    contact_regs = None

    def __init__(self, rawdatadir, patient, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(self.__class__.__name__, self.config.out.FOLDER_LOGS)

        self.rawdatadir = rawdatadir

        # set directories for the datasets 
        self.seegdir = os.path.join(self.rawdatadir, patient, 'seeg', 'fif')
        self.elecdir = os.path.join(self.rawdatadir, patient, 'elec')
        self.dwidir = os.path.join(self.rawdatadir, patient, 'dwi')
        self.tvbdir = os.path.join(self.rawdatadir, patient, 'tvb')
        
        self._renamefiles()
        self._loadmetadata()

    def _loadmetadata(self, loadsurf=False):
        self.logger.debug('Reading in metadata!')
        # rename files from .xyz -> .txt
        self._renamefiles()
        self._loadseegxyz()
        self._mapcontacts_toregs()

        # load in ez hypothesis and connectivity from TVB pipeline
        self._loadezhypothesis()
        self._loadconnectivity()

        # also load in surface data
        if loadsurf:
            self._loadsurface()

        self.logger.debug("Finished reading in metadata!")

    def _renamefiles(self):
        sensorsfile = os.path.join(self.elecdir, 'seeg.xyz')
        newsensorsfile = os.path.join(self.elecdir, 'seeg.txt')
        gainfile = os.path.join(self.elecdir, 'gain_inv-square.mat')
        newgainfile = os.path.join(self.elecdir, 'gain_inv-square.txt')
        try:
            os.rename(sensorsfile, newsensorsfile)
        except:
            self.logger.debug("\nAlready renamed seeg.xyz possibly!\n")
        try:
            os.rename(gainfile, newgainfile)
        except:
            self.logger.debug("\nAlready renamed gain.mat possibly!\n")
     
        self.sensorsfile = newsensorsfile
        self.gainfile = os.path.join(self.elecdir, 'gain_inv-square.txt')
        if not os.path.exists(self.sensorsfile):
            self.gainfile = os.path.join(self.elecdir, 'gain_inv-square.dk.txt')

    def _loadseegxyz(self):
        seeg_pd = utils.loadseegxyz(self.sensorsfile)
        self.chanxyz_labels = np.array(seeg_pd.index.values)
        self.chanxyz = seeg_pd.as_matrix(columns=None)
        self.logger.debug("\nLoaded in seeg xyz coords!\n")

    def _mapcontacts_toregs(self):
        contacts_file = os.path.join(self.elecdir, 'seeg.txt')
        self.label_volume_file = os.path.join(self.dwidir, 'label_in_T1.nii.gz')
        if not os.path.exists(self.label_volume_file):
            self.label_volume_file = os.path.join(self.dwidir, 'label_in_T1.dk.nii.gz')
        self.contact_regs = np.array(utils.mapcontacts_toregs(contacts_file, self.label_volume_file))
        self.logger.debug("\nMapped contacts to regions!\n")

    def _loadezhypothesis(self):
        self.ez_hyp_file = os.path.join(self.tvbdir, 'ez_hypothesis.txt')
        if not os.path.exists(self.ez_hyp_file):
            self.ez_hyp_file = os.path.join(self.tvbdir, 'ez_hypothesis.dk.txt')

        reginds = pd.read_csv(self.ez_hyp_file, delimiter='\n').as_matrix()
        self.ezinds = np.where(reginds==1)[0]
        self.logger.info("\nLoaded in ez hypothesis!\n")

    def _loadgain(self):
        pass

    def _loadconnectivity(self):
        self.connfile = os.path.join(self.tvbdir, 'connectivity.zip')
        if not os.path.exists(self.connfile):
            self.connfile = os.path.join(self.tvbdir, 'connectivity.dk.zip')
            
        conn_loader = LoadConn()
        conn = conn_loader.readconnectivity(self.connfile)
        self.conn = conn
        self.logger.info("\nLoaded in connectivity!\n")

    def _loadsurface(self):
        self.surf = LoadSurface()
        self.verts, self.normals, self.areas, self.regmap = self.surf.loadsurfdata(self.tvbdir, use_subcort=False)

