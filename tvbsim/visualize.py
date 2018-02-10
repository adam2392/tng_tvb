import os.path
import numpy as np
from matplotlib import colors, cm
from matplotlib import pyplot as plt
import seaborn as sns

import scipy 
import util
from sklearn.preprocessing import MinMaxScaler

def normalizetime(ts, overallmax=None):
    if not overallmax:
        tsrange = (np.max(ts, 1) - np.min(ts, 1))
    else:
        tsrange = (overallmax - np.min(ts,1))
    ts = ts/tsrange[:,np.newaxis]
    return ts
def normalizeseegtime(ts):
    tsrange = (np.max(ts, 1) - np.min(ts, 1))
    ts = ts/tsrange[:,np.newaxis]

    avg = np.mean(ts, axis=1)
    ts  = ts - avg[:, np.newaxis]
    return ts
def minmaxts(ts):
    scaler = MinMaxScaler()
    return scaler.fit_transform(ts)

def highpassfilter(seegts):
    # seegts = seegts.T
    b, a = scipy.signal.butter(5, 0.5, btype='highpass', analog=False, output='ba')
    seegf = np.zeros(seegts.shape)

    numcontacts, _ = seegts.shape
    for i in range(0, numcontacts):
        seegf[i,:] = scipy.signal.filtfilt(b, a, seegts[i, :])

    return seegf

def randselectindices(allindices, plotsubset, ezindices=[], pzindices=[]):
    # get random indices not within ez, or pz
    numbers = np.arange(0, len(allindices), dtype=int)
    # print ezindices
    numbers = np.delete(numbers, ezindices)
    numbers = np.delete(numbers, pzindices)
    randindices = np.random.choice(numbers, 6)

    if plotsubset:
        indicestoplot = np.array((), dtype='int')
        indicestoplot = np.append(indicestoplot, ezindices)
        indicestoplot = np.append(indicestoplot, pzindices)
        indicestoplot = np.append(indicestoplot, randindices)
    else:
        indicestoplot = np.arange(0,len(allindices), dtype='int')

    return indicestoplot

def selectindices(alllabels, toplotlabels):
    toplotindices = [i for i, x in enumerate(alllabels) if x in toplotlabels]
    return np.array(toplotindices)

'''
Module Object: Plotter / RawPlotter
Description: This is the objects used for grouping plotting under similar code.

These plots can plot z ts, epi ts, seeg ts, and brain hemisphere with regions plotted.
This will help visualize the raw data time series and the locations of seeg within 
brain hemispheres.
'''
class Plotter():
    def __init__(self, axis_font, title_font):
        self.axis_font = axis_font
        self.title_font = title_font

class RawPlotter(Plotter):
    def __init__(self, axis_font=None, title_font=None, color_new=None, figsize=None):
        if not axis_font:
            axis_font = {'family':'Arial', 'size':'30'}

        if not title_font:
            ### Set the font dictionaries (for plot title and axis titles)
            title_font = {'fontname':'Arial', 'size':'30', 'color':'black', 'weight':'normal',
          'verticalalignment':'bottom'} # Bottom vertical alignment for more space

        if not color_new:
            color_new = ['peru', 'dodgerblue', 'slategrey', 
             'skyblue', 'springgreen', 'fuchsia', 'limegreen', 
             'orangered',  'gold', 'crimson', 'teal', 'blueviolet', 'black', 'cyan', 'lightseagreen',
             'lightpink', 'red', 'indigo', 'mediumorchid', 'mediumspringgreen']
        Plotter.__init__(self, axis_font, title_font)
        # self.initializefig(figsize)
        self.color_new = color_new

    def initializefig(self, figsize=None):
        sns.set_style("darkgrid")
        self.fig = plt.figure(figsize=figsize)
        self.axes = plt.gca()

    def plotzts(self, zts, onsettimes=[], offsettimes=[]):
        self.axes.plot(zts.squeeze(), color='black')
        self.axes.set_title('Z Region Time Series', **self.title_font)
        
        # adapt the axis fonts for this plot
        plt.rc('font', **self.axis_font)
        
        # plot onset/offset times predicted from the z ts
        for i in range(len(onsettimes)):
            self.axes.axvline(onsettimes[i])
        for i in range(len(offsettimes)):
            self.axes.axvline(offsettimes[i])
            
        self.fig.tight_layout()
        plt.show()
        
        return self.fig

    def plotepileptorts(self, epits, region_labels, ezregion, pzregion, times, onsettimes=[], offsettimes=[], plotsubset=False):
        '''
        Function for plotting the epileptor time series for a given patient

        Can also plot a subset of the time series.

        Performs normalization along each channel separately. 
        '''
        print "ezreion is: ", ezregion
        print "pzregion is: ", pzregion
        print "time series shape is: ", epits.shape
        
        # get the indices for ez and pz region
        # initialize object to assist in moving seeg contacts
        movecontact = util.MoveContacts([], [], region_labels, [], True)
        ezindices = movecontact.getindexofregion(ezregion)
        pzindices = movecontact.getindexofregion(pzregion)

        # define specific regions to plot
        regionstoplot = randselectindices(region_labels, plotsubset, ezindices, pzindices)

        # get shapes of epits
        numregions, numsamps = epits.shape
        # locations to plot for each plot along y axis
        regf = 0; regt = len(regionstoplot)
        # get the time window range to plot
        timewindowbegin = 0; timewindowend = numsamps

        timestoplot = times[timewindowbegin:timewindowend]

        # get the epi ts to plot and the corresponding time indices
        epitoplot = epits[regionstoplot, timewindowbegin:timewindowend]

        # Normalize the time series in the time axis to have nice plots
        # overallmax = np.max(epitoplot.ravel())
        # epitoplot = normalizetime(epitoplot, overallmax)
        epitoplot = normalizetime(epitoplot)
        ######################### PLOTTING OF EPILEPTOR TS ########################
        # plot time series
        # times[timewindowbegin:timewindowend], 
        self.axes.plot(epitoplot.T + np.r_[regf:regt], 'k')

        # plot 3 different colors - normal, ez, pz
        colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
        colors = ['red','blue', 'black']
        for i,j in enumerate(self.axes.lines):
            if i in ezindices:
                j.set_color(colors[0])
            elif i in pzindices:
                j.set_color(colors[1])
            else:
                j.set_color(colors[2])

        # plot vertical lines of 'predicted' onset/offset
        for idx in range(0, len(onsettimes)):
            self.axes.axvline(onsettimes[idx], color='red', linestyle='dashed')
        for idx in range(0, len(offsettimes)):
            self.axes.axvline(offsettimes[idx], color='red', linestyle='dashed')
        
        # adapt the axis fonts for this plot
        plt.rc('font', **self.axis_font)
        self.axes.set_title('Epileptor TVB Simulated TS for ' + ' nez=' + str(len(ezregion)) + ' npz='+ str(len(pzregion)), **self.title_font)
        self.axes.set_xlabel('Time (msec)')
        self.axes.set_ylabel('Regions in Parcellation N=84')
        self.axes.set_yticks(np.r_[regf:regt])
        self.axes.set_yticklabels(region_labels[regionstoplot])
        try:
            self.fig.tight_layout()
        except:
            print "can't tight layout"
        plt.show()

        return self.fig

    def plotseegts(self, seegts, chanlabels, times, onsettimes, 
                    offsettimes, plotsubset=False):
        '''
        Function for plotting the epileptor time series for a given patient

        Can also plot a subset of the time series.

        Performs normalization along each channel separately. 
        '''
        # get shapes of epits
        numchans, numsamps = seegts.shape

        # get the time window range to plot
        timewindowbegin = 0
        timewindowend = numsamps

        # get the channels to plot indices
        chanstoplot = randselectindices(chanlabels, plotsubset, ezindices=[], pzindices=[])
        chanstoplot = chanstoplot.astype(int)

        # locations to plot for each plot along y axis
        regf = 0; regt = len(chanstoplot)
        reg = np.linspace(0, (len(chanstoplot)+1)*2, len(chanstoplot)+1)
        reg = reg[0:]
        
        # get the epi ts to plot and the corresponding time indices
        seegtoplot = seegts[chanstoplot, timewindowbegin:timewindowend]
        timestoplot = times[timewindowbegin:timewindowend]
            
        # Normalize the time series in the time axis to have nice plots also high pass filter
        # overallmax = np.max(seegtoplot.ravel())
        # seegts = normalizetime(seegtoplot, overallmax)
        seegtoplot = normalizetime(seegtoplot)
        seegtoplot = seegtoplot - np.mean(seegtoplot, axis=1)[:, np.newaxis]
        ######################### PLOTTING OF SEEG TS ########################
        plottedts = seegtoplot.T 
        yticks = np.nanmean(seegtoplot, axis=1, dtype='float64')

        # plot time series
        self.axes.plot(seegtoplot.T + np.r_[regf:regt], 
                             color='black', linewidth=3)
        
        # plot 3 different colors - normal, ez, pz
        colors = ['red','blue', 'black']

        # adapt the axis fonts for this plot
        plt.rc('font', **self.axis_font)
        
        # plot vertical lines of 'predicted' onset/offset
        for idx in range(0, len(onsettimes)):
            self.axes.axvline(onsettimes[idx], color='red', linestyle='dashed')
        for idx in range(0, len(offsettimes)):
            self.axes.axvline(offsettimes[idx], color='red', linestyle='dashed')

        # self.axes.set_title('SEEG TVB Simulated TS for ' + patient + ' nez=' + str(len(ezregion)) + ' npz='+ str(len(pzregion)), **self.title_font)            
        self.axes.set_xlabel('Time (msec)')
        self.axes.set_ylabel('Channels N=' + str(len(chanlabels)))
        self.axes.set_yticks(np.r_[regf:regt])
        self.axes.set_yticklabels(chanlabels[chanstoplot])
        # self.fig.tight_layout()
        # plt.show()

        return self.fig

    def plotseegchans(self, seegts, times, chanlabels, onsettimes, 
                    offsettimes, toplotlabels=[]):
        # get shapes of epits
        numchans, numsamps = seegts.shape

        # get the time window range to plot
        timewindowbegin = 0
        timewindowend = numsamps

        # get the channels to plot indices
        chanstoplot = selectindices(chanlabels, toplotlabels)
        chanstoplot = chanstoplot.astype(int)

        # hard coded modify
        # chanstoplot = [11, 12, 13, 15, 16, 17]
        # locations to plot for each plot along y axis
        regf = 0; regt = len(chanstoplot)
        reg = np.linspace(0, (len(chanstoplot)+1)*2, len(chanstoplot)+1)
        reg = reg[0:]
        # regt = len(regionstoplot)

        # Normalize the time series in the time axis to have nice plots also high pass filter
        # seegts = highpassfilter(seegts)
        # overallmax = np.max(seegts.ravel())
        # seegts = normalizetime(seegts, overallmax)
        seegts = normalizetime(seegts)
        
        # get the epi ts to plot and the corresponding time indices
        seegtoplot = seegts[chanstoplot, timewindowbegin:timewindowend]
        timestoplot = times[timewindowbegin:timewindowend]

        seegtoplot = seegtoplot - np.mean(seegtoplot, axis=1)[:, np.newaxis]
            
        ######################### PLOTTING OF SEEG TS ########################
        plottedts = seegtoplot.T 
        yticks = np.nanmean(seegtoplot, axis=1, dtype='float64')

        # plot time series
        self.axes.plot(timestoplot, seegtoplot.T + np.r_[regf:regt], 
                             color='black', linewidth=3)
        
        # adapt the axis fonts for this plot
        plt.rc('font', **self.axis_font)
        
        # plot vertical lines of 'predicted' onset/offset
        for idx in range(0, len(onsettimes)):
            self.axes.axvline(onsettimes[idx], color='red', linestyle='dashed')
        for idx in range(0, len(offsettimes)):
            self.axes.axvline(offsettimes[idx], color='red', linestyle='dashed')

        self.axes.set_title('SEEG TVB Simulated TS ', **self.title_font)            
        self.axes.set_xlabel('Time (msec)')
        self.axes.set_ylabel('Channels N=' + str(len(chanlabels)))
        self.axes.set_yticks(np.r_[regf:regt])
        # self.axes.set_yticks(yticks)
        self.axes.set_yticklabels(chanlabels[chanstoplot])
        self.fig.tight_layout()
        plt.show()

        return self.fig

    def plotregions(self, xreg, yreg, numregions):
        # divide into equal regions for left/right hemisphere
        self.axes.plot(xreg[0:numregions//2], yreg[0:numregions//2], 'ro')
        #and black for Right Hemisphere
        self.axes.plot(xreg[numregions//2:] , yreg[numregions//2:], 'ko')
    def plotlabeledregion(self, xreg, yreg, ezindices, label, color='blue'):
        self.axes.plot(xreg[ezindices] , yreg[ezindices], color=color, marker='o', linestyle="None", markersize=12, label=label)  ### EZ

    def plotcontactsinbrain(self, cort_surf, regioncentres, regionlabels, seeg_xyz, seeg_labels, incr_cont, patient, ezindices, pzindices=[]):
        # get xyz coords of centres
        xreg, yreg, zreg = regioncentres.T
        numregions = int(regioncentres.shape[0])
        
        numcontacts = seeg_xyz.shape[0]

        # get the number of contacts
        nCols_new = len(incr_cont)
        
        # SEEG location as red 
        xs, ys, zs = seeg_xyz.T # SEEG coordinates --------> (RB)'s electrodes concatenated

        x_cort, y_cort, z_cort = cort_surf.vertices.T
        V = pzindices
        U = ezindices
        # V = []
        
        ii=0
        
        print "num regions: ", numregions
        print "num contacts: ", numcontacts
        print nCols_new
        print "xreg: ", xreg.shape
        print "yreg: ", yreg.shape
        print "zreg: ", zreg.shape
        print U
        print V
        
        # Plot the regions along their x,y coordinates
        self.plotregions(xreg, yreg, numregions)
        # Plot the ez region(s)
        self.plotlabeledregion(xreg, yreg, ezindices, label='EZ', color='red')
        # Plot the pz region(s)
        self.plotlabeledregion(xreg, yreg, pzindices, label='PZ', color='blue')
        
        #################################### Plot surface vertices  ###################################    
        self.axes.plot(x_cort, y_cort, alpha=0.2) 
        contourr = -4600
        self.axes.plot(x_cort[: contourr + len(x_cort)//2], y_cort[: contourr + len(x_cort)//2], 'gold', alpha=0.1) 
        
        #################################### Elecrodes Implantation  ###################################    
        # plot the contact points
        self.axes.plot(xs[:incr_cont[ii]], ys[:incr_cont[ii]], 
                  self.color_new[ii] , marker = 'o', label= seeg_labels[ii])

        # add label at the first contact for electrode
        self.axes.text(xs[0], ys[0],  str(seeg_labels[ii]), color = self.color_new[ii], fontsize = 20)

        for ii in range(1,nCols_new):
            self.axes.plot(xs[incr_cont[ii-1]:incr_cont[ii]], ys[incr_cont[ii-1]:incr_cont[ii]], 
                 self.color_new[ii] , marker = 'o', label= seeg_labels[incr_cont[ii-1]])
            self.axes.text(xs[incr_cont[ii-1]], ys[incr_cont[ii-1]],  
                str(seeg_labels[incr_cont[ii-1]]), color = self.color_new[ii], fontsize = 20)

        for er in range(numregions):
            self.axes.text(xreg[er] , yreg[er] + 0.7, str(er+1), color = 'g', fontsize = 15)

        self.axes.set_xlabel('x')
        self.axes.set_ylabel('y')
        self.axes.set_title('SEEG Implantations for ' + patient + 
            ' nez=' + str(len(ezindices)) + ' npz='+ str(len(pzindices)), **self.title_font)            
        self.axes.grid(True)
        self.axes.legend()
        plt.show()

        return self.fig

if __name__ == '__main__':
    print "hi"