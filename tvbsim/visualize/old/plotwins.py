import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from .basevisual import BaseVisualModel


class VisualWins(BaseVisualModel):
    def __init__(self, figsize=(7, 7), title_font=[], axis_font=[]):
        BaseVisualModel.__init__(self, title_font, axis_font)
        self.figsize = figsize
        # super(BaseVisualModel, self).__init__(self, title_font, axis_font)

    def loadtimewins(self, timewins):
        self.timewins = timewins

    def _converttimestowindow(self, times):
        winindices = []
        if times.size > 1:
            for time in times:
                timeindex = np.where(np.logical_and(self.timewins[:, 0] < time,
                                                    self.timewins[:, 1] > time))[0]
                if len(timeindex) >= 1:
                    winindices.append(timeindex[0])
        else:
            winindices.append(np.where(np.logical_and(self.timewins[:, 0] < times,
                                                      self.timewins[:, 1] > times))[0][0])
        return winindices

    def plotvertwins(self, onsettimes=np.array([]), offsettimes=np.array([])):
        onsettimes = np.array(onsettimes)
        offsettimes = np.array(offsettimes)

        onsetinds = self._converttimestowindow(onsettimes)
        offsetinds = self._converttimestowindow(offsettimes)
        self.plotvertlines(onsetinds, offsetinds)

    def heatwinmodel(self, xsubset=False, cbarlab=None, normalize=True):
        numchans, numsamps = self.data.shape

        # get the channels to plot indices
        xindices = np.arange(0, numsamps).astype(int)
        # get the ylabel indices to plot
        if xsubset:
            pass
        else:
            timebegin = 0
            timeend = numsamps

        # get the epi ts to plot and the corresponding time indices
        datatoplot = self.data[:, timebegin:timeend]

        ######################### PLOTTING OF HEATMAP ########################
        # plot time series
        self.fig = plt.figure(figsize=self.figsize)
        self.plots = sns.heatmap(data=datatoplot, yticklabels=np.flipud(self.ylabels),
                                 cmap=plt.cm.jet, cbar=True, cbar_kws={'label': cbarlab})
        self.ax = plt.gca()
        # self.ax.invert_yaxis()
        # adapt the axis fonts for this plot
        plt.rc('font', **self.axis_font)

        titlestr = 'Fragility Heatmap'
        self.ax.set_title(titlestr, **self.title_font)
        self.ax.set_xlabel('Time (msec)')
        self.ax.set_ylabel('Channels N=' + str(len(self.ylabels)))
        return self.fig, self.ax
