from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger
from tvbsim.base.utils.data_structures_utils import NumpyEncoder

import numpy as np
import json
import io
import os
try:
    to_unicode = unicode
except NameError:
    to_unicode = str
    
''' Class wrappers for writing HPC mvar model computations '''
class BaseHPC(object):
    ''' Required attribbutes when running a job array '''
    numtasks = None
    taskid = None 
    tempdir = None
    numcores = None 

    def __init__(self, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

    def computenumwins(self):
        msg = "Base HPC model method is not implemented."
        raise NotImplementedError(msg)

    def loaddata(self):
        msg = "Base HPC model method is not implemented."
        raise NotImplementedError(msg)

    def loadmetafile(self):
        msg = "Base HPC model method is not implemented."
        raise NotImplementedError(msg)

    def run(self):
        msg = "Base HPC model method is not implemented."
        raise NotImplementedError(msg)
  
    def loadmetadata(self, metadata):
        self.metadata = metadata

    def _loadnpzfile(self, npzfilename):
        if npzfilename.endswith('.npz'):
            npzfilename = npzfilename.split('.npz')[0]
        result = np.load(npzfilename)
        self.logger.debug("loaded npzfilename: {} with keys: {}".format(npzfilename, self.result.keys()))
        return result 

    def _writenpzfile(self, npzfilename, **kwargs):
        self.logger.debug("saved npzfilename: ", npzfilename)
        self.logger.debug("saved: ", kwargs.keys())
        np.savez_compressed(npzfilename, **kwargs)

    def _writejsonfile(self, metadata, metafilename):
        if not metafilename.endswith('.json'):
            metafilename += '.json'
        with io.open(metafilename, mode='w', encoding='utf8') as outfile:
            str_ = json.dumps(metadata, 
                                indent=4, sort_keys=True, 
                                cls=NumpyEncoder, separators=(',', ': '), 
                                ensure_ascii=False)
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
                json_str = fp.read()
            metadata = json.loads(json_str)
        return metadata