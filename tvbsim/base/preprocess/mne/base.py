from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger
import numpy as np

class BaseFreqModel(object):
    def __init__(self, winsize, stepsize, samplerate, config=None):
        self.config = config or Config()
        self.logger = initialize_logger(
            self.__class__.__name__,
            self.config.out.FOLDER_LOGS)

        self.winsize = winsize
        self.stepsize = stepsize
        self.samplerate = samplerate

    def compress_windata(data, samplepoints):
        '''
        Compresses the data with a window/stepsize specified by user
        over the sample points
        '''
        numchans, numfreqs, _ = data.shape
        numwins = len(samplepoints)
        compressed_data = np.zeros((numchans, numfreqs, numwins))

        # loop through each window and compress the data with average
        for iwin in range(numwins):
            compressed_data[:,:,iwin] = np.mean(data[:,:,samplepoints[iwin,0]:samplepoints[iwin,1]], axis=-1)
        return compressed_data
