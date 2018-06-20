import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from .basevisual import BaseVisualModel

class VisualTs(BaseVisualModel):
    def __init__(self, figsize=(7, 7), title_font=[], axis_font=[]):
        BaseVisualModel.__init__(self, title_font, axis_font)
        self.figsize = figsize

    def colorts(self, indices, color='red'):
        '''
        '''
        if not isinstance(indices, list):
            try:
                indices = list(indices)
            except TypeError:
                indices = [indices]
        # plot colors for lines
        for i, j in enumerate(self.ax.lines):
            if i in indices:
                j.set_color(color)

    def plotsinglets(self, ts):
        # plot time series
        self.fig = plt.figure(figsize=self.figsize)
        self.plots = plt.plot(ts,
                              color='black', linewidth=3)
        self.ax = plt.gca()
        # adapt the axis fonts for this plot
        plt.rc('font', **self.axis_font)
        titlestr = 'Z Region Time Series'
        self.ax.set_title(titlestr, **self.title_font)
        self.ax.set_xlabel('Time (msec)')
        self.ax.set_ylabel('Value of Z Variable')

        return self.fig, self.ax

    def plotts(self, ysubset=False, xsubset=False,
               normalize=True, titlestr=None):
        numchans, numsamps = self.data.shape

        # get the channels to plot indices
        yindices = np.arange(0, len(self.ylabels)).astype(int)
        xindices = np.arange(0, numsamps).astype(int)
        # get the ylabel indices to plot
        if ysubset is True:
            ylabs_toplot = self._randselectindices(
                indices=yindices, numtoselect=numtoselect)
        elif ysubset is list:
            ylabs_toplot = self._selectindices(ysubset)
        else:
            ylabs_toplot = yindices
        if xsubset:
            pass
        else:
            timebegin = 0
            timeend = numsamps
        # locations to plot for each plot along y axis
        regf = 0
        regt = len(ylabs_toplot)

        # get the epi ts to plot and the corresponding time indices
        datatoplot = self.data[ylabs_toplot, timebegin:timeend]
        # Normalize the time series in the time axis to have nice plots also
        # high pass filter
        if normalize:
            overallmax = np.max(datatoplot.ravel())
            overallmax = False
            datatoplot = self._normalizets(datatoplot, overallmax)
            # datatoplot = self._normalizets(datatoplot)
            # datatoplot = datatoplot - np.mean(datatoplot, axis=1)[:, np.newaxis]

        ######################### PLOTTING OF SEEG TS ########################
        yticks = np.nanmean(datatoplot, axis=1, dtype='float64')

        # plot time series
        self.fig = plt.figure(figsize=self.figsize)
        self.plots = plt.plot(datatoplot.T + np.r_[regf:regt],
                              color='black', linewidth=3)
        self.ax = plt.gca()
        # adapt the axis fonts for this plot
        plt.rc('font', **self.axis_font)

        if titlestr is None:
            titlestr = 'SEEG Time Series Simulated'
        self.ax.set_title(titlestr, **self.title_font)
        self.ax.set_xlabel('Time (msec)')
        self.ax.set_ylabel('Channels N=' + str(len(self.ylabels)))
        self.ax.set_yticks(np.r_[regf:regt])
        self.ax.set_yticklabels(self.ylabels[ylabs_toplot])
        return self.fig, self.ax
