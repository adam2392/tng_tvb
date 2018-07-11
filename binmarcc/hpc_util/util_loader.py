import os
import sys
import numpy as np
import json
sys.path.append('../')

from tvbsim.io.loaders.perfile.loadtvbdata import StructuralDataLoader

def load_data(patient, rawdatadir):
    if patient not in rawdatadir:
        rawdatadir = os.path.join(rawdatadir, patient)
    loader = StructuralDataLoader(root_dir=rawdatadir)
    return loader

def load_sim_data(patient, rawdatadir):
    datafilename = recording.split('/')[-1].split('.')[0].split(patient)[-1]
    rawdata, metadata = loader.load_dataset(idatafile,  
                                        reference=reference,
                                        clip=False, 
                                        sync=True, recording_type='ictal')

    return rawdata, metadata