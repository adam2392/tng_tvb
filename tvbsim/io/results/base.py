import os
import json
import io
import zipfile
import numpy as np

from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger

# to allow compatability between python2/3
try:
    to_unicode = unicode
except NameError:
    to_unicode = str


class BaseResultLoader(object):
    result = None
    chanlabels = []

    def __init__(self, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
                        self.__class__.__name__,
                        self.config.out.FOLDER_LOGS)

    def _exists(self, filepath):
        return os.path.exists(filepath)

    def _init_files(self):
        '''
        Initialization function to be called
        '''
        if self.name not in self.resultsdir:
            self.resultsdir = os.path.join(self.resultsdir, self.name)
            
    def _writejsonfile(self, metadata, metafilename):
        with io.open(metafilename, 'w', encoding='utf8') as outfile:
            str_ = json.dumps(metadata,
                              indent=4, sort_keys=True, cls=NumpyEncoder,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(to_unicode(str_))

    def _loadjsonfile(self, metafilename):
        # if not metafilename.endswith('.json'):
        #     metafilename += '.json'

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

    def _loadnpzfile(self, npzfilename):
        if npzfilename.endswith('.json'):
            npzfilename = npzfilename.split('.json')[0]
        self.result = np.load(npzfilename)
        print("loaded npzfilename: {} with keys: {}".format(npzfilename, self.result.keys()))

    def _writenpzfile(self, npzfilename, kwargs):
        print("saved npzfilename: ", npzfilename)
        print("saved: ", kwargs.keys())
        np.savez_compressed(npzfilename, **kwargs)

    def _check_all_files(self):
        print("Checking root pat directory of results: ", self._exists(self.resultsdir))
        print("Files in this results directory are: ", os.listdir(self.resultsdir))

