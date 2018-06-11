import os
import zipfile

import numpy as np
from tvbsim.io.results.base import BaseResultLoader

from tvbsim.base.dataobjects.freqmap import Freqmap
from tvbsim.base.dataobjects.heatmap import Heatmap
from tvbsim.base.dataobjects.winmat import Winmat 

ANALYSIS_TYPES = ['pert', 'mvar', 'fft', 'morlet', 'img']
ICTAL_TYPES = ['ictal', 'sz', 'seiz', 'seizure']
INTERICTAL_TYPES = ['interictal', 'ii', 'aslp', 'aw']

class SubjectResults(BaseResultLoader):
    mvar_results = []
    pert_results = []
    fft_results = []
    morlet_results = []
    filepaths = []
    jsonfiles = []
    def __init__(self, name, resultsdir=None, atlas=None, DEBUG=True, config=None):
        super(SubjectResults, self).__init__(config=config)
        self.name = name
        self.resultsdir = resultsdir
        if atlas is not None:
            self.atlas = atlas

        # initializations - to find files
        self._init_files()
        if DEBUG:
            self._check_all_files()

        self.load_file_paths()

    def _get_file_name(self, metadata, analysis_type, json_file):
        try:
            if analysis_type == 'mvar':
                filename = os.path.join(self.resultsdir, metadata['mvarfilename'])
            elif analysis_type == 'pert':
                filename = os.path.join(self.resultsdir, metadata['pertfilename'])
            elif analysis_type == 'fft':
                filename = os.path.join(self.resultsdir, metadata['fftfilename'])
            elif analysis_type == 'morlet':
                filename = os.path.join(self.resultsdir, metadata['morletfilename'])
        except KeyError as e:
            print("Key error accessing filename: ", e)
            if analysis_type == 'mvar':
                filename = os.path.join(self.resultsdir, json_file.split('meta')[0]+ 'mvarmodel.npz')
            elif analysis_type == 'pert':
                filename = os.path.join(self.resultsdir, json_file.split('meta')[0]+ 'pertmodel.npz')
            elif analysis_type == 'fft':
                filename = os.path.join(self.resultsdir, json_file.split('meta')[0]+ 'model.npz')
            elif analysis_type == 'morlet':
                filename = os.path.join(self.resultsdir, json_file.split('meta')[0]+ 'model.npz')
        return filename

    def load_file_paths(self):
        '''
        Load all the json files to get the filepaths for the corresponding
        results.
        '''
        json_files = [filename for filename in os.listdir(self.resultsdir) \
                if filename.endswith('.json') if not filename.startswith('.')]

        # for buggy code that saved as .npz
        json_files = [filename for filename in os.listdir(self.resultsdir) \
                if 'meta' in filename if not filename.startswith('.')]

        for json_file in json_files:
            json_file = os.path.join(self.resultsdir, json_file)
            metadata = self._loadjsonfile(json_file)

            self.jsonfiles.append(json_file)
            if 'mvar' in json_file:                
                filename = self._get_file_name(metadata, analysis_type='mvar', json_file=json_file)
                self.filepaths.append(filename)
            elif 'pert' in json_file:
                filename = self._get_file_name(metadata, analysis_type='pert',  json_file=json_file)
                self.filepaths.append(filename)
            elif 'fft' in json_file:
                filename = self._get_file_name(metadata, analysis_type='fft', json_file=json_file)
                self.filepaths.append(filename)
            elif 'morlet' in json_file:
                filename = self._get_file_name(metadata, analysis_type='morlet',  json_file=json_file)
                self.filepaths.append(filename)
            else:
                raise ValueError("Unexpected result type: %s" % json_file)


    def read_results(self, type, freqbandname = 'GAMMA'):
        '''
        Actually load in all the results into a list for user ot work with
        '''
        json_files = [filename for filename in os.listdir(self.resultsdir) \
                if filename.endswith('.json') if not filename.startswith('.')]

        for json_file in json_files:
            metadata = self._loadjsonfile(json_file)
            chanlabels = metadata['chanlabels']
            locations = metadata['chanxyz']
            timepoints = metadata['timepoints']

            if 'mvar' in json_file:                
                mvarfilename = self._get_file_name(metadata, analysis_type='mvar', json_file=json_file)
                self._loadnpzfile(mvarfilename)
                adjmats = self.result['adjmats']
                result = Winmat(mvarfilename, win_matrix=adjmats, 
                                labels=chanlabels, locations=locations, timepoints=timepoints)
                self.mvar_results.append(result)
            elif 'pert' in json_file:
                pertfilename = self._get_file_name(metadata, analysis_type='pert', json_file=json_file)
                self._loadnpzfile(pertfilename)
                pertmats = self.result['pertmats']
                result = Heatmap(pertfilename, win_matrix=pertmats, 
                                labels=chanlabels, locations=locations, timepoints=timepoints)
                self.pert_results.append(result)
            elif 'fft' in json_file:
                filename = self._get_file_name(metadata, analysis_type='fft', json_file=json_file)
                self._loadnpzfile(filename)
                power = self.result['power']
                freqs = metadata['freqs']
                result = Heatmap(filename, win_matrix=power, 
                                labels=chanlabels, locations=locations, 
                                timepoints=timepoints, freqs=freqs,
                                freqbandname=freqbandname)
                self.fft_results.append(result)
            elif 'morlet' in json_file:
                filename = self._get_file_name(metadata, analysis_type='morlet', json_file=json_file)
                self._loadnpzfile(filename)
                power = self.result['power']
                freqs = metadata['freqs']
                result = Heatmap(filename, win_matrix=power, 
                                labels=chanlabels, locations=locations, 
                                timepoints=timepoints, freqs=freqs,
                                freqbandname=freqbandname)
                self.morlet_results.append(result)
            else:
                raise ValueError("Unexpected result type: %s" % json_file)

    def read_result(self, json_file, freqbandname='GAMMA'):
        '''
        Read in one result at a time from their crresponding json file
        '''
        metadata = self._loadjsonfile(json_file)
        chanlabels = metadata['chanlabels']
        locations = metadata['chanxyz']
        timepoints = metadata['timepoints']

        if 'mvar' in json_file:                
            mvarfilename = self._get_file_name(metadata, analysis_type='mvar', json_file=json_file)
            self._loadnpzfile(mvarfilename)
            adjmats = self.result['adjmats']
            result = Winmat(mvarfilename, win_matrix=adjmats, 
                            labels=chanlabels, locations=locations, timepoints=timepoints)
        elif 'pert' in json_file:
            pertfilename =  self._get_file_name(metadata, analysis_type='pert', json_file=json_file)
            self._loadnpzfile(pertfilename)
            pertmats = self.result['pertmats']
            result = Heatmap(pertfilename, win_matrix=pertmats, 
                            labels=chanlabels, locations=locations, timepoints=timepoints)
        elif 'fft' in json_file:
            filename =  self._get_file_name(metadata, analysis_type='fft', json_file=json_file)
            self._loadnpzfile(filename)
            power = self.result['power']
            freqs = metadata['freqs']
            result = Freqmap(filename, win_matrix=power, 
                            labels=chanlabels, locations=locations, 
                            timepoints=timepoints, freqs=freqs,
                            freqbandname=freqbandname)
        elif 'morlet' in json_file:
            filename =  self._get_file_name(metadata, analysis_type='morlet', json_file=json_file)
            self._loadnpzfile(filename)
            power = self.result['power']
            freqs = metadata['freqs']
            result = Freqmap(filename, win_matrix=power, 
                            labels=chanlabels, locations=locations, 
                            timepoints=timepoints, freqs=freqs,
                            freqbandname=freqbandname)
        else:
            raise ValueError("Unexpected result type: %s" % json_file)

        return result




