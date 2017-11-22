import os.path
import numpy as np
from matplotlib import colors, cm
import time
import scipy 
from matplotlib import pyplot as plt

def getindexofregion(regions, ezregion=[], pzregion=[]):
    sorter = np.argsort(regions)
    ezindices = sorter[np.searchsorted(regions, ezregion, sorter=sorter)]
    pzindices = sorter[np.searchsorted(regions, pzregion, sorter=sorter)]

    return ezindices, pzindices

def highpassfilter(seegts):
    b, a = scipy.signal.butter(2, 0.1, btype='highpass', output='ba')
    seegf = np.zeros(seegts.shape)

    numcontacts, _ = seegts.shape
    for i in range(0, numcontacts):
        seegf[i,:] = scipy.signal.filtfilt(b, a, seegtf[i, :])

    return seegf

def plotepileptorts(epits, regions, ezregion, pzregion=[]):
    # initialize figure
    epifig = plt.figure(figsize=(9,10))

    # get shapes of epits
    numregions, numsamps = epits.shape

    # get the time window range to plot
    timewindowbegin = 0
    timewindowend = numsamps

    # get the ez/pz indices
    ezindices, pzindices = getindexofregion(regions, ezregion, pzregion)
    
    # get random indices not within ez, or pz
    numbers = np.arange(0, numregions)
    numbers = np.delete(numbers, np.concatenate((ezindices, pzindices), axis=0))
    randindices = np.random.choice(numbers, 3)
    
    # create the indices of regions to plot
    regionstoplot = np.array((), dtype='int')
    regionstoplot = np.append(regionstoplot, ezindices)
    regionstoplot = np.append(regionstoplot, pzindices)
    regionstoplot = np.append(regionstoplot, randindices)
    regionstoplot = np.arange(0,len(regions), dtype='int')

    # Normalize the time series in the time axis to have nice plots
    epirange = (np.max(epits, 1) - np.min(epits, 1))
    epits = epits/epirange[:,np.newaxis]

    # get the epi ts to plot and the corresponding time indices
    epitoplot = epits[regionstoplot, timewindowbegin:timewindowend]
    timestoplot = times[timewindowbegin:timewindowend]

    regionlabels = regions[regionstoplot]

    # regularization factors for each plot
    regf = 0
    regt = len(regionstoplot)


        
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

    for idx in range(0, len(onsettimes)):
        plt.axvline(onsettimes[idx], color='red')
        plt.axvline(offsettimes[idx], color='red')
        
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Regions in Parcellation N=84')
    ax.set_title('Epileptor TVB Simulated TS')
    ax.set_yticks(np.r_[regf:regt])
    ax.set_yticklabels(regions[regionstoplot])
    plt.tight_layout()
    plt.show()


    return epifig

def plotseegts(seegts, seegxyz, regioncentres, ezregion, pzregion=None):
    # get 2-3 electrodes closest to ezregion

    # get 2-3 electrodes closest to pzregion
    pass

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