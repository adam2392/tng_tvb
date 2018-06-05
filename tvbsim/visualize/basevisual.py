import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class BaseVisualModel(object):
    def __init__(self, title_font=[], axis_font=[]):
        if isinstance(title_font, list):
            if not title_font:
                # Set the font dictionaries (for plot title and axis titles)
                self.title_font = {'fontname': 'Arial',
                                   'size': '34',
                                   'color': 'black',
                                   'weight': 'normal',
                                   'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
        if isinstance(axis_font, list):
            if not axis_font:
                self.axis_font = {'family': 'Arial',
                                  'size': '30'}

    def setfigsize(self, figsize):
        self.figsize = figsize

    def loaddata(self, data, ylabels=np.array([])):
        '''

        '''
        assert data.shape[0] <= data.shape[1]
        self.data = data
        # if not isinstance(ylabels, list):
        #     ylabels = list(ylabels)
        if ylabels.size == 0:
            warnings.warn("User needs to pass in the list of ylabels!")
        else:
            self.ylabels = ylabels

    def plotvertlines(self, onsettimes=[], offsettimes=[]):
        '''
        '''
        # plot vertical lines of 'predicted' onset/offset
        for idx in range(0, len(onsettimes)):
            self.ax.axvline(onsettimes[idx], color='green', linestyle='dashed')
        for idx in range(0, len(offsettimes)):
            self.ax.axvline(offsettimes[idx], color='red', linestyle='dashed')

    def _normalizets(self, data, overallmax=False):
        '''
        '''
        if not overallmax:
            tsrange = (np.max(data, 1) - np.min(data, 1))
        else:
            tsrange = (overallmax - np.min(data, 1))
        ts = data / tsrange[:, np.newaxis]
        return ts

    def _randselectindices(self, indices, numtoselect=6):
        '''

        '''
        # get a random selection of the indices
        randindices = np.random.choice(indices, numtoselect).astype(int)
        return randindices

    def _selectindices(self, toplotylabels):
        '''
        '''
        toplotindices = np.array([i for i, y in enumerate(self.ylabels)
                                  if y in toplotylabels])
        return toplotindices
