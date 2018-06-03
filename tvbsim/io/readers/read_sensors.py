import numpy as np 
import zipfile
import scipy.io
import os
import pandas as pd
from fragility.base.dataobjects.connectivity import Connectivity
from fragility.base.dataobjects.sensors import Sensors
from fragility.base.utils.log_error import initialize_logger
from fragility.base.utils.data_structures_utils import ensure_list
import h5py

from enum import Enum
'''

Load regions functions for fragility analysis module to get structural
connectivity data and load it so that you can get access to

- region labels in cortex
- region xyz coords
- weights between regions (derived from dMRI)
- and much more...
'''

class LoadSensors(object):
    logger = initialize_logger(__name__)

    def readsensors(self, path):
        """
        :param path: Path towards a custom head folder
        :return: 3 lists with all sensors from Path by type
        """
        self.logger.info("Starting to read Sensors data from: %s" % path)

        seegfile = os.path.join(path, 'seeg.txt')
        invgainfile = os.path.join(path, 'gain_inv-square.txt')
        # dipgainfile = os.path.join(path, 'gain_dipole_no-subcort.mat')

        labels, locations = self.loadseegxyz(seegfile)
        gainmat = self.loadgainmat(invgainfile)

        # self.logger.debug("Labels: %s" % labels)

        sensors = Sensors(labels, locations=locations, gain_matrix=gainmat)
        return sensors

    def loadseegxyz(self, seegfile):
        '''
        This is just a wrapper function to retrieve the seeg coordinate data in a pd dataframe
        '''
        seeg_pd = pd.read_csv(
            seegfile, names=['x', 'y', 'z'], delim_whitespace=True)
        self.seegfile = seegfile
        self.seeg_labels = seeg_pd.index.values.tolist()
        self.seeg_xyz = seeg_pd.as_matrix(columns=None)

        return self.seeg_labels, self.seeg_xyz

    def loadgainmat(self, gainfile):
        # function to get model in its equilibrium value
        gain_pd = pd.read_csv(gainfile, header=None, delim_whitespace=True)
        self.gainfile = gainfile
        self.gainmat = gain_pd.as_matrix()
        return self.gainmat
    