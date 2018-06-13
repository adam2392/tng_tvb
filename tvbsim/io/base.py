import os
import numpy as np
import pandas as pd
import json
from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger
from tvbsim.io.utils import utils, seegrecording
from tvbsim.io.readers.read_connectivity import LoadConn
import io
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
        if not os.path.exists(self.gainfile):
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
        tvb_sourcefile = os.path.join(self.tvbdir, 'connectivity.zip')
        if not os.path.exists(tvb_sourcefile):
            tvb_sourcefile = os.path.join(self.tvbdir, 'connectivity.dk.zip')

        conn_loader = LoadConn()
        conn = conn_loader.readconnectivity(tvb_sourcefile)
        self.conn = conn
        self.logger.info("\nLoaded in connectivity!\n")

    def _writejsonfile(self, metafilename):
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

        self.metadata = metadata
        return self.metadata

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

    def convert_raw_mne(self, rawdata, info):
        # info = mne.create_info(ch_names, sfreq)
        raw = mne.io.RawArray(rawdata, info)
        self.raw = raw

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
