import os
import numpy as np
import pandas as pd
import json
from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger
from tvbsim.io.utils import utils, seegrecording
from tvbsim.io.readers.read_connectivity import LoadConn
import io

# to allow compatability between python2/3
try:
    to_unicode = unicode
except NameError:
    to_unicode = str
class BaseLoader(object):
    # gainfile = None
    # sensorsfile = None
    # connfile = None
    # surfacefile = None
    # label_volume_file = None

    # elec_dir = None
    # seegdir = None
    # dwidir = None
    # tvbdir = None
    def __init__(self, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

    def _renamefiles(self):
        sensorsfile = os.path.join(self.elecdir, 'seeg.xyz')
        newsensorsfile = os.path.join(self.elecdir, 'seeg.txt')
        try:
            # copyfile(sensorsfile, newsensorsfile)
            os.rename(sensorsfile, newsensorsfile)
        except BaseException:
            self.logger.debug("\nAlready renamed seeg.xyz possibly!\n")

        self.sensorsfile = newsensorsfile
        self.gainfile = os.path.join(self.elecdir, 'gain_inv-square.txt')
        if not os.path.exists(self.sensorsfile):
            self.gainfile = os.path.join(self.elecdir, 'gain_inv-square.dk.txt')

    def _loadseegxyz(self):
        seegfile = os.path.join(self.elecdir, 'seeg.txt')
        seeg_pd = utils.loadseegxyz(seegfile)
        self.chanxyz_labels = np.array(seeg_pd.index.values)
        self.chanxyz = seeg_pd.as_matrix(columns=None)
        self.logger.debug("\nLoaded in seeg xyz coords!\n")

    def _mapcontacts_toregs(self):
        contacts_file = os.path.join(self.elecdir, 'seeg.txt')
        self.label_volume_file = os.path.join(
            self.dwidir, 'label_in_T1.nii.gz')
        if not os.path.exists(self.label_volume_file):
            self.label_volume_file = os.path.join(
                self.dwidir, 'label_in_T1.dk.nii.gz')
        self.contact_regs = np.array(
            utils.mapcontacts_toregs(
                contacts_file,
                self.label_volume_file))
        self.logger.debug("\nMapped contacts to regions!\n")

    def _loadezhypothesis(self):
        ez_file = os.path.join(self.tvbdir, 'ez_hypothesis.txt')
        if not os.path.exists(ez_file):
            ez_file = os.path.join(self.tvbdir, 'ez_hypothesis.dk.txt')

        reginds = pd.read_csv(ez_file, delimiter='\n').as_matrix()
        self.ezinds = np.where(reginds == 1)[0]
        self.logger.info("\nLoaded in ez hypothesis!\n")

    def _loadconnectivity(self):
        self.connfile = os.path.join(self.tvbdir, 'connectivity.zip')
        if not os.path.exists(self.connfile):
            self.connfile = os.path.join(self.tvbdir, 'connectivity.dk.zip')

        conn_loader = LoadConn()
        conn = conn_loader.readconnectivity(self.connfile)
        self.conn = conn
        self.logger.info("\nLoaded in connectivity!\n")


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
            with open(metafilename, 'r', encoding='utf8') as f:
                metadata = json.load(f)
        except:
            with io.open(metafilename, encoding='utf-8', mode='r') as fp:
                json_str = fp.read() #json.loads(
            metadata = json.loads(json_str)

        self.metadata = metadata
