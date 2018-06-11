import os
import numpy as np
import pandas as pd
import json
import io
import sys
from enum import Enum

from tvbsim.io.base import BaseLoader
from tvbsim.base.dataobjects.freqmap import Freqmap
from tvbsim.base.dataobjects.winmat import Winmat
from tvbsim.base.dataobjects.heatmap import Heatmap

from tvbsim.base.constants.config import Config
from tvbsim.base.utils.log_error import initialize_logger
import tvbsim.base.constants.model_constants as constants
from tvbsim.base.utils.data_structures_utils import ensure_list

class ResultTypes(Enum):
    TYPE_FRAGILITY = 'FRAGILITY'
    TYPE_POWERSPECT = "POWER_SPECTRUM"
    TYPE_STATE = 'STATE_TRANSITION_MATRIX'

class LoadResults(BaseLoader):
    TYPE_FRAGILITY = ResultTypes.TYPE_FRAGILITY
    TYPE_POWERSPECT = ResultTypes.TYPE_POWERSPECT
    TYPE_STATE = ResultTypes.TYPE_STATE

    timepoints = None
    freqs = None
    chanlabels = None
    locations = None

    onsetind = None
    offsetind = None

    def __init__(self, root_dir, datafile, metadatafile,
                 h_type=TYPE_FRAGILITY.value, patient=None,
                 freqbandname=constants.GAMMA, config=None):
        super(LoadResults, self).__init__(config=config)

        self.root_dir = root_dir
        self.datafile = datafile
        self.metadatafile = metadatafile
        self.patient = patient
        self.h_type = h_type

        self.freqbandname = freqbandname
        # load in the data
        self.load_data()
        try:
            self.load_szinds()
        except Exception as e:
            print(e)
            print("Can't load in seizure indices")

    def _findtimewins(self, times):
        indices = []
        for time in ensure_list(times):
            if time == 0:
                indices.append(time)
            else:
                idx = (time >= self.timepoints[:,0])*(time <= self.timepoints[:,1])
                timeind = np.where(idx)[0]
                if len(timeind) > 0:
                    indices.append(timeind[0])
                else:
                    indices.append(np.nan)
        return indices

    def load_szinds(self):
        # onsettimes = self.metadata['onsetind']
        # offsettimes = self.metadata['offsetind']
        onsettimes = self.metadata['onsettimes']
        offsettimes = self.metadata['offsettimes']
        self.timepoints = np.array(self.timepoints)

        # get the actual indices that occur within time windows
        if onsettimes is not None:
            self.onsetind = self._findtimewins(onsettimes)
        if offsettimes is not None:
            self.offsetind = self._findtimewins(offsettimes)

        print(self.timepoints[-1,:])
        print(self.onsetind, self.offsetind)

    def _loaddata_frommeta(self):
        # extract useful metadata
        if self.h_type == self.TYPE_POWERSPECT:
            self.freqs = self.metadata['freqs']
        self.labels = self.metadata['chanlabels']

        # this is done in case the metadata did not save the timepoints
        # in future, this would be deprecated as all data would be stored
        # in the metafile
        try:    
            self.timepoints = self.metadata['timepoints']
            self.locations = self.metadata['chanxyz']
        except:
            self.timepoints = self.resultstruct['metadata'].item()['timepoints']


    def load_data(self):
        datafilepath = os.path.join(self.root_dir, self.datafile)
        self.logger.debug(
            "The data file to read is {}".format(datafilepath))
        self.resultstruct = self._loadnumpystruct(datafilepath)

        self.metadata = self.resultstruct['metadata'].item()
        # load metadata via .json object if avail.
        # metadatafilepath = os.path.join(self.root_dir, self.metadatafile)
        # self.logger.debug(
        #     "The meta data file to use is {}".format(metadatafilepath))
        # self._loadjsonfile(metadatafilepath)
        self._loaddata_frommeta()

        if self.h_type == self.TYPE_FRAGILITY.value:
            self.pertmat = self.resultstruct['pertmats']
            self.delvecs = self.resultstruct['delvecs']

            self.result = Heatmap(datafilepath, self.pertmat,
                                  labels=self.labels,
                                  locations=self.locations,
                                  timepoints=self.timepoints,)
        elif self.h_type == self.TYPE_POWERSPECT.value:
            self.power = self.resultstruct['power']
            self.phase = self.resultstruct['phase']

            self.result = Freqmap(datafilepath, self.power,
                                  labels=self.labels,
                                  locations=self.locations,
                                  timepoints=self.timepoints,
                                  freqs=self.freqs,
                                  freqbandname=self.freqbandname)
        elif self.h_type == self.TYPE_STATE.value:
            self.adjmats = self.resultstruct['adjmats']

            self.result = Winmat(datafilepath, self.adjmats,
                                 labels=self.labels,
                                 timepoints=self.timepoints)

    def _loadnumpystruct(self, datafilepath):
        if not datafilepath.endswith('.npz'):
            datafilepath += '.npz'
        self.resultstruct = np.load(datafilepath, encoding='latin1')
        return self.resultstruct

    def getmetadata(self):
        """
        If the user wants to clip the data, then you can save a separate metadata
        file that contains all useful metadata about the dataset.
        """
        return self.metadata
