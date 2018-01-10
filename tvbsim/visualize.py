import os.path
import numpy as np
from matplotlib import colors, cm
from matplotlib import pyplot as plt
import seaborn as sns

import scipy 
import util
from sklearn.preprocessing import MinMaxScaler

def normalizetime(ts):
    tsrange = (np.max(ts, 1) - np.min(ts, 1))

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

def defineindicestoplot(allindices, plotsubset, ezindices=[], pzindices=[]):
    # get random indices not within ez, or pz
    numbers = np.arange(0, len(allindices), dtype=int)
    # print ezindices
    print numbers
    numbers = np.delete(numbers, ezindices)
    numbers = np.delete(numbers, pzindices)
    randindices = np.random.choice(numbers, 3)

    if plotsubset:
        indicestoplot = np.array((), dtype='int')
        indicestoplot = np.append(indicestoplot, ezindices)
        indicestoplot = np.append(indicestoplot, pzindices)
        indicestoplot = np.append(indicestoplot, randindices)
    else:
        indicestoplot = np.arange(0,len(allindices), dtype='int')
    print "here:", indicestoplot
    return indicestoplot

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
        self.initializefig(figsize)
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
            self.axes.axvline(offsettimes[i])
            
        self.fig.tight_layout()
        plt.show()
        
        return self.fig

    def plotepileptorts(self, epits, times, metadata, onsettimes, offsettimes, patient='', plotsubset=False):
        '''
        Function for plotting the epileptor time series for a given patient

        Can also plot a subset of the time series.

        Performs normalization along each channel separately. 
        '''
        # extract metadata
        region_labels = metadata['regions']
        region_centers = metadata['regions_centers']
        seeg_labels = metadata['seeg_contacts']
        seeg_xyz = metadata['seeg_xyz']
        ezregion = metadata['ez']
        pzregion = metadata['pz']
        ezindices = metadata['ezindices']
        pzindices = metadata['pzindices']
        x0ez = metadata['x0ez']
        x0pz = metadata['x0pz']
        x0norm = metadata['x0norm']

        print "ezreion is: ", ezregion
        print "pzregion is: ", pzregion
        print "x0 values are (ez, pz, norm): ", x0ez, x0pz, x0norm
        print "time series shape is: ", epits.shape
        
        # get the indices for ez and pz region
        # initialize object to assist in moving seeg contacts
        movecontact = util.MoveContacts(seeg_labels, seeg_xyz, region_labels, region_centers, True)
        ezindices = movecontact.getindexofregion(ezregion)
        pzindices = movecontact.getindexofregion(pzregion)

        # define specific regions to plot
        regionstoplot = defineindicestoplot(region_labels, plotsubset, ezindices, pzindices)
        print regionstoplot
        # get shapes of epits
        numregions, numsamps = epits.shape
        # locations to plot for each plot along y axis
        regf = 0; regt = len(regionstoplot)
        # get the time window range to plot
        timewindowbegin = 0; timewindowend = numsamps

        timestoplot = times[timewindowbegin:timewindowend]

        # Normalize the time series in the time axis to have nice plots
        epits = normalizetime(epits)

        # get the epi ts to plot and the corresponding time indices
        epitoplot = epits[regionstoplot, timewindowbegin:timewindowend]
            
        ######################### PLOTTING OF EPILEPTOR TS ########################
        # plot time series
        self.axes.plot(times[timewindowbegin:timewindowend], epitoplot.T + np.r_[regf:regt], 'k')

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
        try:
            for idx in range(0, len(onsettimes)):
                self.axes.axvline(onsettimes[idx], color='red', linestyle='dashed')
                self.axes.axvline(offsettimes[idx], color='red', linestyle='dashed')
        except Exception as e:
            print e
            print "Was trying to plot the onset/offset times for epileptor ts!"

        # adapt the axis fonts for this plot
        plt.rc('font', **self.axis_font)
        self.axes.set_title('Epileptor TVB Simulated TS for ' + patient + ' nez=' + str(len(ezregion)) + ' npz='+ str(len(pzregion)), **self.title_font)
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

    def plotseegts(self, seegts, times, metadata, onsettimes, offsettimes, ezseegindex=[], patient='', plotsubset=False):
        '''
        Function for plotting the epileptor time series for a given patient

        Can also plot a subset of the time series.

        Performs normalization along each channel separately. 
        '''
        # extract metadata
        regions = metadata['regions']
        ezregion = metadata['ez']
        pzregion = metadata['pz']
        ezindices = metadata['ezindices']
        pzindices = metadata['pzindices']
        x0ez = metadata['x0ez']
        x0pz = metadata['x0pz']
        x0norm = metadata['x0norm']
        chanlabels = metadata['seeg_contacts']

        # get shapes of epits
        numchans, numsamps = seegts.shape

        # get the time window range to plot
        timewindowbegin = 0
        timewindowend = numsamps

        print "ez seeg index is: ", ezseegindex
        # get the channels to plot indices
        chanstoplot = defineindicestoplot(chanlabels, plotsubset, ezindices=ezseegindex, pzindices=[])
        chanstoplot = chanstoplot.astype(int)

        # hard coded modify
        # chanstoplot = [11, 12, 13, 15, 16, 17]
        # locations to plot for each plot along y axis
        # locations to plot for each plot along y axis
        regf = 0; regt = len(chanstoplot)
        reg = np.linspace(0, (len(chanstoplot)+1)*2, len(chanstoplot)+1)
        reg = reg[0:]
        # regt = len(regionstoplot)

        # Normalize the time series in the time axis to have nice plots also high pass filter
        # seegts = highpassfilter(seegts)
        seegts = normalizetime(seegts)
        
        # get the epi ts to plot and the corresponding time indices
        seegtoplot = seegts[chanstoplot, timewindowbegin:timewindowend]
        timestoplot = times[timewindowbegin:timewindowend]

        seegtoplot = seegtoplot - np.mean(seegtoplot, axis=1)[:, np.newaxis]
            
        ######################### PLOTTING OF SEEG TS ########################
        plottedts = seegtoplot.T 
        yticks = np.nanmean(seegtoplot, axis=1, dtype='float64')

        # print np.mean(seegts[14,:])
        # print type(seegtoplot[0,0])
        # print type(seegtoplot)
        # print plottedts.shape
        # print yticks
        # plot time series
        self.axes.plot(timestoplot, seegtoplot.T + np.r_[regf:regt], 
                             color='black', linewidth=3)
        
        # plot 3 different colors - normal, ez, pz
        colors = ['red','blue', 'black']
        # if plotsubset: # plotting subset of all the seeg channels
        #     for idx, chan in enumerate(chanstoplot):
        #         if chan == ezseegindex:
        #             self.axes.plot(timestoplot, seegtoplot[idx,:].T + reg[idx], 
        #                      color='red', linewidth=3)
        #         else:
        #             self.axes.plot(timestoplot, seegtoplot[idx,:].T + reg[idx], 
        #                      color='black', linewidth=3)
        # else:   # plotting all the seeg channels
        #     for idx, chan in enumerate(chanstoplot):
        #         if chan == ezseegindex:
        #             self.axes.plot(timestoplot, seegtoplot[chan,:].T + reg[chan], 
        #                      color='red', linewidth=3)
        #         else:
        #             self.axes.plot(timestoplot, seegtoplot[chan,:].T + reg[chan], 
        #                      color='black', linewidth=3)

        # adapt the axis fonts for this plot
        plt.rc('font', **self.axis_font)
        
        # plot vertical lines of 'predicted' onset/offset
        for idx in range(0, len(onsettimes)):
            self.axes.axvline(onsettimes[idx], color='red', linestyle='dashed')
            self.axes.axvline(offsettimes[idx], color='red', linestyle='dashed')

        self.axes.set_title('SEEG TVB Simulated TS for ' + patient + ' nez=' + str(len(ezregion)) + ' npz='+ str(len(pzregion)), **self.title_font)            
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
    def plotlabeledregion(self, xreg, yreg, ezindices, label):
        self.axes.plot(xreg[ezindices] , yreg[ezindices], 'bo', markersize=12, label=label)  ### EZ

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
        self.plotlabeledregion(xreg, yreg, ezindices, label='EZ')
        # Plot the pz region(s)
        self.plotlabeledregion(xreg, yreg, pzindices, label='PZ')
        
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

# def plotepileptorts(epits, times, metadata, patient, plotsubset=False):
#     '''
#     Function for plotting the epileptor time series for a given patient

#     Can also plot a subset of the time series.

#     Performs normalization along each channel separately. 
#     '''
#     sns.set_style("darkgrid")

#     # extract metadata
#     region_labels = metadata['regions']
#     region_centers = metadata['regions_centers']
#     seeg_labels = metadata['seeg_contacts']
#     seeg_xyz = metadata['seeg_xyz']
#     ezregion = metadata['ez']
#     pzregion = metadata['pz']
#     ezindices = metadata['ezindices']
#     pzindices = metadata['pzindices']
#     x0ez = metadata['x0ez']
#     x0pz = metadata['x0pz']
#     x0norm = metadata['x0norm']

#     print "ezreion is: ", ezregion
#     print "pzregion is: ", pzregion
#     print "x0 values are (ez, pz, norm): ", x0ez, x0pz, x0norm
#     print "time series shape is: ", epits.shape
    
#     # determine onset/offset times
#     postprocessor = util.PostProcess(epits, seegts, times)
#     onsettimes, offsettimes = postprocessor.findonsetoffset(zts[ezindices, :].squeeze())

#     # get the indices for ez and pz region
#     # initialize object to assist in moving seeg contacts
#     movecontact = util.MoveContacts(seeg_labels, seeg_xyz, region_labels, region_centers, True)
#     ezindices = movecontact.getindexofregion(ezregion)
#     pzindices = movecontact.getindexofregion(pzregion)
    
#     # get shapes of epits
#     numregions, numsamps = epits.shape

#     # get the time window range to plot
#     timewindowbegin = 0; timewindowend = numsamps

#     # define specific regions to plot
#     regionstoplot = defineregionstoplot(regions, plotsubset, ezindices, pzindices)
#     regionlabels = region_labels[regionstoplot]
    
#     # locations to plot for each plot along y axis
#     regf = 0
#     regt = len(regionstoplot)

#     # Normalize the time series in the time axis to have nice plots
#     epits = normalizetime(epits)

#     # get the epi ts to plot and the corresponding time indices
#     epitoplot = epits[regionstoplot, timewindowbegin:timewindowend]
#     timestoplot = times[timewindowbegin:timewindowend]
        
#     ######################### PLOTTING OF EPILEPTOR TS ########################
#     # initialize figure
#     epifig = plt.figure(figsize=(9,7))
#     # plot time series
#     epilines = plt.plot(timestoplot, epitoplot.T + np.r_[regf:regt], 'k')
#     ax = plt.gca()

#     # plot 3 different colors - normal, ez, pz
#     colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
#     colors = ['red','blue', 'black']
#     for i,j in enumerate(ax.lines):
#         if i in ezindices:
#             j.set_color(colors[0])
#         elif i in pzindices:
#             j.set_color(colors[1])
#         else:
#             j.set_color(colors[2])

#     # plot vertical lines of 'predicted' onset/offset
#     try:
#         for idx in range(0, len(onsettimes)):
#             plt.axvline(onsettimes[idx], color='red', linestyle='dashed')
#             plt.axvline(offsettimes[idx], color='red', linestyle='dashed')
#     except Exception as e:
#         print e
#         print "Was trying to plot the onset/offset times for epileptor ts!"

#     ax.set_xlabel('Time (msec)')
#     ax.set_ylabel('Regions in Parcellation N=84')
#     ax.set_title('Epileptor TVB Simulated TS for ' + patient + ' nez=' + str(len(ezregion)) + ' npz='+ str(len(pzregion)))
#     ax.set_yticks(np.r_[regf:regt])
#     ax.set_yticklabels(region_labels[regionstoplot])
#     plt.tight_layout()
#     plt.show()

#     return epifig, ax

# def plotzts(zts, ezloc, onsettimes, offsettimes):
#     ### Set the font dictionaries (for plot title and axis titles)
#     title_font = {'fontname':'Arial', 'size':'34', 'color':'black', 'weight':'normal',
#               'verticalalignment':'bottom'} # Bottom vertical alignment for more space
#     axis_font = {'family':'Arial', 'size':'30'}

#     # plot the z ts
#     fig = plt.figure()
#     plt.plot(zts, color='black')
#     plt.title('Z Region Time Series', **title_font)
#     plotax = plt.gca()
    
#     # adapt the axis fonts for this plot
#     plt.rc('font', **axis_font)
    
#     # plot onset/offset times predicted from the z ts
#     for i in range(len(onsettimes)):
#         plt.axvline(onsettimes[i])
#         plt.axvline(offsettimes[i])
        
#     plt.tight_layout()
#     plt.show()
    
#     return fig, plotax

# def plotseegts(seegts, times, metadata, onsettimes, offsettimes,
#                patient, ezseegindex, plotsubset=False):
#     '''
#     Function for plotting the epileptor time series for a given patient

#     Can also plot a subset of the time series.

#     Performs normalization along each channel separately. 
#     '''
#     # extract metadata
#     regions = metadata['regions']
#     ezregion = metadata['ez']
#     pzregion = metadata['pz']
#     ezindices = metadata['ezindices']
#     pzindices = metadata['pzindices']
#     x0ez = metadata['x0ez']
#     x0pz = metadata['x0pz']
#     x0norm = metadata['x0norm']
#     chanlabels = metadata['seeg_contacts']

#     ### Set the font dictionaries (for plot title and axis titles)
#     title_font = {'fontname':'Arial', 'size':'34', 'color':'black', 'weight':'normal',
#               'verticalalignment':'bottom'} # Bottom vertical alignment for more space
#     axis_font = {'family':'Arial', 'size':'18'}

#     # get shapes of epits
#     numchans, numsamps = seegts.shape

#     # get the time window range to plot
#     timewindowbegin = 0
#     timewindowend = numsamps

#     # get the channels to plot indices
#     chanstoplot = defineindicestoplot(chanlabels, plotsubset, ezseegindex, pzindices=[])

#     # locations to plot for each plot along y axis
#     reg = np.linspace(0, (len(chanstoplot)+1)*2, len(chanstoplot)+1)
#     reg = reg[1:]

#     # Normalize the time series in the time axis to have nice plots also high pass filter
#     # seegts = highpassfilter(seegts)
#     seegts = normalizetime(seegts)

#     print "ezreion is: ", ezregion
#     print "pzregion is: ", pzregion
#     print "x0 values are (ez, pz, norm): ", x0ez, x0pz, x0norm
#     print "time series shape is: ", seegts.shape
#     print "ez seeg index is: ", ezseegindex
#     print "chanstoplot are: ", chanstoplot
#     print reg

#     # get the epi ts to plot and the corresponding time indices
#     seegtoplot = seegts[chanstoplot, timewindowbegin:timewindowend]
#     timestoplot = times[timewindowbegin:timewindowend]
#     #     seegtoplot = seegtoplot - np.mean(seegtoplot, axis=1)[:, np.newaxis]
        
#     ######################### PLOTTING OF SEEG TS ########################
#     # initialize figure
#     seegfig = plt.figure(figsize=(17,15))
#     # plot time series
#     seeglines = plt.plot(timestoplot, seegtoplot.T + reg[:len(chanstoplot)], 
#                          color='black', linewidth=3)
#     ax = plt.gca()
    
#     # plot 3 different colors - normal, ez, pz
#     colors = ['red','blue', 'black']
#     #     seeglines = ax.get_lines()
#     if plotsubset:
#         for idx, chan in enumerate(chanstoplot):
#             print idx, chan
#             if chan == ezseegindex:
#                 print "here"
#                 print np.mean(seegtoplot[idx,:])
#                 plt.plot(timestoplot, seegtoplot[idx,:].T + reg[idx], 
#                          color='red', linewidth=3)
#             else:
#                 print "over here"
#                 print np.mean(seegtoplot[idx,:])
#                 plt.plot(timestoplot, seegtoplot[idx,:].T + reg[idx], 
#                          color='black', linewidth=3)
#     else:
#         for idx, chan in enumerate(chanstoplot):
#             print idx, chan
#             if chan == ezseegindex:
#                 print "here"
#                 print np.mean(seegtoplot[chan,:])
#                 plt.plot(timestoplot, seegtoplot[chan,:].T + reg[chan], 
#                          color='red', linewidth=3)
#             else:
#                 print "over here"
#                 print np.mean(seegtoplot[chan,:])
#                 plt.plot(timestoplot, seegtoplot[chan,:].T + reg[chan], 
#                          color='black', linewidth=3)

#     # adapt the axis fonts for this plot
#     plt.rc('font', **axis_font)
    
#     # plot vertical lines of 'predicted' onset/offset
#     for idx in range(0, len(onsettimes)):
#         plt.axvline(onsettimes[idx], color='red', linestyle='dashed')
#         plt.axvline(offsettimes[idx], color='red', linestyle='dashed')
        
#     ax.set_xlabel('Time (msec)')
#     ax.set_ylabel('Channels N=' + str(len(chanlabels)))
#     ax.set_title('SEEG TVB Simulated TS for ' + patient + 
#                  ' nez=' + str(len(ezregion)) + ' npz='+ str(len(pzregion)), **title_font)
#     ax.set_yticks(reg)
#     ax.set_yticklabels(chanlabels[chanstoplot])
#     plt.tight_layout()
#     plt.show()

#     return seegfig

# def plotcontactsinbrain(cort_surf, regioncentres, regionlabels, seeg_xyz, seeg_labels, incr_cont, ezindices, pzindices=None):
#     # get xyz coords of centres
#     xreg, yreg, zreg = regioncentres.T
#     numregions = int(regioncentres.shape[0])
    
#     numcontacts = seeg_xyz.shape[0]
#     # get the number of contacts
#     nCols_new = len(incr_cont)
    
#     # SEEG location as red 
#     xs, ys, zs = seeg_xyz.T # SEEG coordinates --------> (RB)'s electrodes concatenated

#     x_cort, y_cort, z_cort = cort_surf.vertices.T
#     V = pzindices
#     U = ezindices
#     V = []
    
#     ii=0
    
    
#     print "num regions: ", numregions
#     print "num contacts: ", numcontacts
#     print nCols_new
#     print "xreg: ", xreg.shape
#     print "yreg: ", yreg.shape
#     print "zreg: ", zreg.shape
#     print U
#     print V
    
#     ### Begin Plotting
#     brainfig = plt.figure(figsize=(10,8))

#     # divide into equal regions for left/right hemisphere
#     plt.plot(xreg[0:numregions//2], yreg[0:numregions//2], 'ro')
#     #and black for Right Hemisphere
#     plt.plot(xreg[numregions//2:] , yreg[numregions//2:], 'ko')

#     #################################### Plot surface vertices  ###################################    
#     plt.plot(x_cort, y_cort, alpha=0.2) 

#     contourr = -4600
#     plt.plot(x_cort[: contourr + len(x_cort)//2], y_cort[: contourr + len(x_cort)//2], 'gold', alpha=0.1) 

#     #################################### label regions EZ ###################################    
#     # plot(xreg[U] , yreg[U], 'bo', markersize=12)  ### EZ
#     plt.plot(xreg[U] , yreg[U], 'bo', markersize=12, label="EZ")  ### EZ

#     #################################### Elecrodes Implantation  ###################################    
#     # plot the contact points
#     plt.plot(xs[:incr_cont[ii]], ys[:incr_cont[ii]], 
#               color_new[ii] , marker = 'o', label= elect[ii])

#     # add label at the first contact for electrode
#     plt.text(xs[0], ys[0],  str(elect[ii]), color = color_new[ii], fontsize = 20)

#     for ii in range(1,nCols_new):
#         plt.plot(xs[incr_cont[ii-1]:incr_cont[ii]], ys[incr_cont[ii-1]:incr_cont[ii]], 
#              color_new[ii] , marker = 'o', label= elect[incr_cont[ii-1]])
#         plt.text(xs[incr_cont[ii-1]], ys[incr_cont[ii-1]],  str(elect[incr_cont[ii-1]]), color = color_new[ii], fontsize = 20)

#     for er in range(numregions):
#         plt.text(xreg[er] , yreg[er] + 0.7, str(er+1), color = 'g', fontsize = 15)

#     plt.xlabel('x')
#     plt.ylabel('y')

#     plt.grid(True)
#     plt.legend()
#     plt.show()

if __name__ == '__main__':
    print "hi"