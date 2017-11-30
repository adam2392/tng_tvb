import os.path
import numpy as np
from matplotlib import colors, cm
import time
import scipy 
from matplotlib import pyplot as plt
# from sklearn.preprocessing import MinMaxScaler

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
    b, a = scipy.signal.butter(2, 0.1, btype='highpass', output='ba')
    seegf = np.zeros(seegts.shape)

    numcontacts, _ = seegts.shape
    for i in range(0, numcontacts):
        seegf[i,:] = scipy.signal.filtfilt(b, a, seegts[i, :])

    return seegf

def plotepileptorts(epits, times, metadata, patient, plotsubset=False):
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

    print "ezreion is: ", ezregion
    print "pzregion is: ", pzregion
    print "x0 values are (ez, pz, norm): ", x0ez, x0pz, x0norm
    print "time series shape is: ", epits.shape
    
    onsettimes = metadata['onsettimes']
    offsettimes = metadata['offsettimes']
    try:
        onsettimes = np.array([tupl for tupl in onsettimes])
        offsettimes = np.array([tupl for tupl in offsettimes])
    except Exception as e:
        print e

    # get shapes of epits
    numregions, numsamps = epits.shape

    # get the time window range to plot
    timewindowbegin = 0
    timewindowend = numsamps

    # get the indices for ez and pz region
    ezindices, pzindices = getindexofregion(regions, ezregion, pzregion)

    # get random indices not within ez, or pz
    numbers = np.arange(0, numregions)
    numbers = np.delete(numbers, np.concatenate((ezindices, pzindices), axis=0))
    randindices = np.random.choice(numbers, 3)

    # define specific regions to plot
    if plotsubset:
        regionstoplot = np.array((), dtype='int')
        regionstoplot = np.append(regionstoplot, ezindices)
        regionstoplot = np.append(regionstoplot, pzindices)
        regionstoplot = np.append(regionstoplot, randindices)
    else:
        regionstoplot = np.arange(0,len(regions), dtype='int')
    regionlabels = regions[regionstoplot]
    # locations to plot for each plot along y axis
    regf = 0
    regt = len(regionstoplot)

    # Normalize the time series in the time axis to have nice plots
    epits = normalizetime(epits)

    # get the epi ts to plot and the corresponding time indices
    epitoplot = epits[regionstoplot, timewindowbegin:timewindowend]
    timestoplot = times[timewindowbegin:timewindowend]
        
    ######################### PLOTTING OF EPILEPTOR TS ########################
    # initialize figure
    epifig = plt.figure(figsize=(9,7))
    # plot time series
    epilines = plt.plot(timestoplot, epitoplot.T + np.r_[regf:regt], 'k')
    ax = plt.gca()

    # plot 3 different colors - normal, ez, pz
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
    colors = ['red','blue', 'black']
    for i,j in enumerate(ax.lines):
        if i in ezindices:
            j.set_color(colors[0])
        elif i in pzindices:
            j.set_color(colors[1])
        else:
            j.set_color(colors[2])

    # plot vertical lines of 'predicted' onset/offset
    # for idx in range(0, len(onsettimes)):
    #     plt.axvline(onsettimes[idx], color='red', linestyle='dashed')
    #     plt.axvline(offsettimes[idx], color='red', linestyle='dashed')
        
    ax.set_xlabel('Time (msec)')
    ax.set_ylabel('Regions in Parcellation N=84')
    ax.set_title('Epileptor TVB Simulated TS for ' + patient + ' nez=' + str(len(ezregion)) + ' npz='+ str(len(pzregion)))
    ax.set_yticks(np.r_[regf:regt])
    ax.set_yticklabels(regions[regionstoplot])
    plt.tight_layout()
    plt.show()

    return epifig

def plotseegts(seegts, times, metadata, patient, ezseegindex, plotsubset=False):
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

    print "ezreion is: ", ezregion
    print "pzregion is: ", pzregion
    print "x0 values are (ez, pz, norm): ", x0ez, x0pz, x0norm
    print "time series shape is: ", seegts.shape
    print "ez seeg index is: ", ezseegindex

    
    onsettimes = metadata['onsettimes']
    offsettimes = metadata['offsettimes']
    try:
        onsettimes = np.array([tupl for tupl in onsettimes])
        offsettimes = np.array([tupl for tupl in offsettimes])
    except Exception as e:
        print e

    # get shapes of epits
    numchans, numsamps = seegts.shape

    # get the time window range to plot
    timewindowbegin = 0
    timewindowend = numsamps

    # get random indices not within ez, or pz
    numbers = np.arange(0, numchans)
    numbers = np.delete(numbers, ezseegindex, axis=0)
    randindices = np.random.choice(numbers, 5)

    # define specific regions to plot
    if plotsubset:
        chanstoplot = np.array((), dtype='int')
        chanstoplot = np.append(chanstoplot, ezseegindex)
        # chanstoplot = np.append(chanstoplot, pzindices)
        chanstoplot = np.append(chanstoplot, randindices)
    else:
        chanstoplot = np.arange(0,len(chanlabels), dtype='int')

    # locations to plot for each plot along y axis
    regf = 0
    regt = len(chanstoplot)

    reg = np.linspace(regf, regt*2, regt)

    # Normalize the time series in the time axis to have nice plots
    # also high pass filter
    # seegts = highpassfilter(seegts)
    seegts = normalizetime(seegts)

    print "regt is: ", regt
    print "chanstoplot are: ", chanstoplot
    print min(seegts[0,:])
    print max(seegts[0,:])

    # get the epi ts to plot and the corresponding time indices
    seegtoplot = seegts[chanstoplot, timewindowbegin:timewindowend]
    timestoplot = times[timewindowbegin:timewindowend]
        
    ######################### PLOTTING OF SEEG TS ########################
    # initialize figure
    seegfig = plt.figure(figsize=(9,5))
    # plot time series
    seeglines = plt.plot(timestoplot, seegtoplot.T + reg[:regt], 'k')
    # seeglines = plt.plot(seegtoplot.T + np.r_[:regt], 'k')
    ax = plt.gca()

    # plot 3 different colors - normal, ez, pz
    colors = ['red','blue', 'black']
    # for i,j in enumerate(ax.lines):
    #     if i == ezseegindex:
    #         j.set_color(colors[0])
    #     # elif i == 0:
    #     #     j.set_color(colors[1])
    #     else:
    #         j.set_color(colors[2])
    if plotsubset:
        # lines = ax.lines
        # lines[0].set_color(colors[0])
        for i,j in enumerate(seeglines):
            if i == 0:
                j.set_color(colors[0])
            elif i==0:
                j.set_color(colors[1])
            else:
                j.set_color(colors[2])
    else:
        for i,j in enumerate(ax.lines):
            if i == ezseegindex:
                print "setting color"
                j.set_color(colors[0])
            elif i == ezseegindex:
                j.set_color(colors[1])
            else:
                j.set_color(colors[2])

    # plot vertical lines of 'predicted' onset/offset
    # for idx in range(0, len(onsettimes)):
    #     plt.axvline(onsettimes[idx], color='red', linestyle='dashed')
    #     plt.axvline(offsettimes[idx], color='red', linestyle='dashed')
        
    ax.set_xlabel('Time (msec)')
    ax.set_ylabel('Channels N=' + str(len(chanlabels)))
    ax.set_title('SEEG TVB Simulated TS for ' + patient + ' nez=' + str(len(ezregion)) + ' npz='+ str(len(pzregion)))
    ax.set_yticks(reg[:regt])
    ax.set_yticklabels(chanlabels[chanstoplot])
    plt.tight_layout()
    plt.show()

    return seegfig

def plotzts(zts, ezloc, onsettimes, offsettimes):
    fig = plt.figure()
    plt.plot(zts[:, ezloc].squeeze())
    plt.title('Z Region Time Series')
    plt.axvline(onsettimes)
    plt.axvline(offsettimes)
    plt.tight_layout()
    plt.show()

def plotcontactsinbrain(regioncentres, cort_surf, seeg_xyz, regionlabels, ezindices, pzindices=None):
    brainfig = plt.figure(figsize=(10,8))

    # get xyz coords of centres
    xreg, yreg, zreg = regioncentres.T

    numregions = regioncentres.shape[0]
    # divide into equal regions for left/right hemisphere
    plt.plot(xreg[:numregions/2] , yreg[:numregions/2], 'ro')
    #and black for Right Hemisphere
    plt.plot(xreg[numregions/2:] , yreg[numregions/2:], 'ko')

    #################################### Plot surface vertices  ###################################    
    x_cort, y_cort, z_cort = cort_surf.vertices.T
    plt.plot(x_cort, y_cort, alpha=0.2) 

    contourr = -4600
    plt.plot(x_cort[: contourr + len(x_cort)//2], y_cort[: contourr + len(x_cort)//2], 'gold', alpha=0.1) 

    #################################### label regions EZ ###################################    
    V = pzindices
    U = ezindices
    print regionlabels[ezindex]

    # plot(xreg[U] , yreg[U], 'bo', markersize=12)  ### EZ
    plot(xreg[U[0]] , yreg[U[0]], 'bo', markersize=12, label="EZ")  ### EZ

    #################################### Elecrodes Implantation  ###################################    
    numcontacts = seeg_xyz.shape[0]
    nCols_new = len(numcontacts)

    # SEEG location as red 
    xs, ys, zs = seeg_xyz.T # SEEG coordinates --------> (RB)'s electrodes concatenated

    ii = 0

    # get number of contacts per electrode
    incr_cont = np.zeros((len(numcontacts)), int)
    incr_cont[0] = 0

    for element_4 in range(0, len(numcontacts)):
        incr_cont[element_4] = incr_cont[element_4-1] + numcontacts[element_4]

    # plot the contact points
    plt.plot(xs[:incr_cont[ii]], ys[:incr_cont[ii]], 
              color_new[ii] , marker = 'o', label= elect[ii])

    # add label at the first contact for electrode
    plt.text(xs[0], ys[0],  str(elect[ii]), color = color_new[ii], fontsize = 20)

        
    for ii in range(1,nCols_new):
        plt.plot(xs[incr_cont[ii-1]:incr_cont[ii]], ys[incr_cont[ii-1]:incr_cont[ii]], 
             color_new[ii] , marker = 'o', label= elect[incr_cont[ii-1]])
        plt.text(xs[incr_cont[ii-1]], ys[incr_cont[ii-1]],  str(elect[incr_cont[ii-1]]), color = color_new[ii], fontsize = 20)

    for er in range(numregions):
        plt.text(xreg[er] , yreg[er] + 0.7, str(er+1), color = 'g', fontsize = 15)

    xlabel('x')
    ylabel('y')

    plt.grid(True)
    plt.legend()
    plt.show()
if __name__ == '__main__':
    print "hi"