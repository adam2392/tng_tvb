import os
import numpy as np
import pandas as pd
import json
import zipfile
from tvbsim.io.utils.elecs import Contacts
from tvbsim.io.readers.read_connectivity import LoadConn
from tvbsim.io.readers.read_surf import LoadSurface 
from tvbsim.io.utils import utils

class MetaLoaders(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def _loadgainmat(self, gainfile):
        if not os.path.exists(gainfile):
            self.logger.error("Can't from {} because doesn't exist".format(gainfile))
            return None
        # function to get model in its equilibrium value
        gain_pd = pd.read_csv(gainfile, header=None, delim_whitespace=True)
        gainmat = gain_pd.as_matrix()
        return gainmat

    def _loadcontacts(self, sensorsfile):
        contacts = Contacts(sensorsfile)
        return contacts

    def _loadseegxyz(self, sensorsfile):
        if not os.path.exists(sensorsfile):
            self.logger.error("Can't from {} because doesn't exist".format(sensorsfile))
            return None

        seeg_pd = utils.loadseegxyz(sensorsfile)
        chanxyzlabels = np.array(seeg_pd.index.values)
        chanxyz = seeg_pd.as_matrix(columns=None)
        self.logger.debug("\nLoaded in seeg xyz coords!\n")
        return chanxyz, chanxyzlabels

    def _mapcontacts_toregs(self, label_volume_file, sensorsfile):
        if not os.path.exists(label_volume_file) or not os.path.exists(sensorsfile):
            self.logger.error("Can't from {} because doesn't exist".format(label_volume_file))
            return None

        contact_regs = np.array(utils.mapcontacts_toregs(
                                    sensorsfile,
                                    label_volume_file))
        self.logger.debug("\nMapped contacts to regions!\n")
        return contact_regs

    def _loadezhypothesis(self, ez_hyp_file):
        if not os.path.exists(ez_hyp_file):
            self.logger.error("Can't from {} because doesn't exist".format(ez_hyp_file))
            return None
            
        reginds = pd.read_csv(ez_hyp_file, header=None, delimiter='\n').as_matrix()
        regezinds = np.where(reginds == 1)[0]
        self.logger.info("\nLoaded in ez hypothesis!\n")
        return regezinds

    def _loadconnectivity(self, connfile):
        if not os.path.exists(connfile):
            self.logger.error("Can't from {} because doesn't exist".format(connfile))
            return None

        conn = LoadConn().readconnectivity(connfile)

        # with zipfile.ZipFile(connfile) as zf:
        #     with zf.open("weights.txt") as fl:
        #         weights = np.genfromtxt(fl, dtype=float)
        #     with zf.open("tract_lengths.txt") as fl:
        #         tract_lengths = np.genfromtxt(fl, dtype=float)
        #     with zf.open("centres.txt") as fl:
        #         region_centres = np.genfromtxt(fl, usecols=(1, 2, 3), dtype=float)
        #     with zf.open("centres.txt") as fl:
        #         region_labels = np.genfromtxt(fl, usecols=(0,), dtype=str)
        return conn 

    def _loadsurface(self, surfacefile, regionmapfile):
        if not os.path.exists(surfacefile) or not os.path.exists(regionmapfile):
            self.logger.error("Can't from {}, {} because doesn't exist".format(surfacefile, regionmapfile))
            return None
        print("loading surface!")
        surf = LoadSurface().loadsurfdata(surfacefile, 
                                        regionmapfile, 
                                        use_subcort=True)
        return surf