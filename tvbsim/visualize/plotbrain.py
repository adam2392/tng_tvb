import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from .basevisual import BaseVisualModel
import re

from ..exp.basetvbexp import TVBExp

class VisualBrain(BaseVisualModel):
    def __init__(self, figsize=(7, 7), title_font=[], axis_font=[], color_new=[]):
        BaseVisualModel.__init__(self, title_font, axis_font)
        self.figsize = figsize

        if len(color_new) == 0:
            self.color_new = ['peru', 'dodgerblue', 'slategrey',
                              'skyblue', 'springgreen', 'fuchsia', 'limegreen',
                              'orangered',  'gold', 'crimson', 'teal', 'blueviolet', 'black', 'cyan', 'lightseagreen',
                              'lightpink', 'red', 'indigo', 'mediumorchid', 'mediumspringgreen']
        else:
            self.color_new = color_new

    def loadsurf(self, vertices, from_file=False):
        self.x_cort, self.y_cort, self.z_cort = vertices.T

    def loadseeg(self, seeg_xyz, seeg_labels, from_file=False):
        # SEEG location as red
        self.xs, self.ys, self.zs = seeg_xyz.T  # SEEG coordinates
        self.seeg_labels = seeg_labels

    def loadregs(self, regioncentres, regionlabels, from_file=False):
        # get xyz coords of centres
        self.xreg, self.yreg, self.zreg = regioncentres.T
        self.regionlabels = regionlabels

    def setcontacts(self):
        elect = []
        dipole = []
        # create lists of elect and dipoles
        for element in range(0, self.seeg_labels.shape[0]):
            kpm = np.array(
                re.match("([A-Z]+[a-z]*[']*)([0-9]+)", self.seeg_labels[element]).groups())
            elect.append(kpm[0])
            dipole.append(int(kpm[1]))

        '''
        Descrip: 
        - Number of electrodes is : len(find_0)
        - Number of contacts per an electrode i is nbr_contacts[i+1]

        '''
        # find the beginning index of each electrode
        find_0 = []
        nbr_contacts = []
        for element_1 in range(0, len(dipole)):
            if dipole[element_1] == 1:
                find_0.append(element_1)
        for element_2 in range(0, len(find_0)-1):
            nbr_contacts.append(find_0[element_2+1]-find_0[element_2])
        nbr_contacts.append(len(self.seeg_labels) - find_0[len(find_0)-1])

        # Find the list of the ending index of each electrode
        incr_cont = np.zeros((len(nbr_contacts)), dtype=int)
        incr_cont[0] = 0

        for element_4 in range(0, len(nbr_contacts)):
            incr_cont[element_4] = incr_cont[element_4-1] + \
                nbr_contacts[element_4]

        self.incr_cont = incr_cont
        print("Nbre_contacts_per_electrode:", nbr_contacts)
        print("Nbre_electrodes:", len(nbr_contacts))
        # To plot each electrode with diff color
        print("Ending Index of electrodes:", incr_cont)

    def plotregions(self):
        numregions = len(self.xreg)
        # divide into equal regions for left/right hemisphere
        self.ax.plot(self.xreg[0:numregions//2],
                     self.yreg[0:numregions//2], 'ro')
        # and black for Right Hemisphere
        self.ax.plot(self.xreg[numregions//2:],
                     self.yreg[numregions//2:], 'ko')

    def plotlabeledregion(self, indices, label, color='blue'):
        if indices.size > 0:
            self.ax.plot(self.xreg[indices],
                         self.yreg[indices],
                         color=color, marker='o',
                         linestyle="None", markersize=12, label=label)  # EZ

    def plotcontactsinbrain(self, ezindices=np.array([], dtype='int'), pzindices=np.array([], dtype='int'), titlestr=None, loc=None):
        if loc is None:
            loc = "upper right"

        numcontacts = len(self.xs)
        numregions = len(self.xreg)
        incr_cont = self.incr_cont
        # get the number of contacts
        nCols_new = len(incr_cont)

        print("num regions: ", numregions)
        print("num contacts: ", numcontacts)
        print(nCols_new)
        print("xreg: ", self.xreg.shape)
        print("yreg: ", self.yreg.shape)
        print("zreg: ", self.zreg.shape)

        # plot The Entire Brain Plot
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = plt.gca()
        # Plot the regions along their x,y coordinates
        self.plotregions()
        # Plot the ez region(s)
        self.plotlabeledregion(ezindices, label='EZ', color='red')
        # Plot the pz region(s)
        self.plotlabeledregion(pzindices, label='PZ', color='blue')

        #################################### Plot surface vertices  ###################################
        self.ax.plot(self.x_cort, self.y_cort, alpha=0.2)
        contourr = -4600
        self.ax.plot(self.x_cort[: contourr + len(self.x_cort)//2],
                     self.y_cort[: contourr + len(self.x_cort)//2], 'gold', alpha=0.1)

        #################################### Elecrodes Implantation  ###################################
        # plot the contact points
        ii = 0
        self.ax.plot(self.xs[:incr_cont[ii]],
                     self.ys[:incr_cont[ii]],
                     self.color_new[ii], marker='o', label=self.seeg_labels[ii])
        # add label at the first contact for electrode
        self.ax.text(self.xs[0], self.ys[0],
                     str(self.seeg_labels[ii]),
                     color=self.color_new[ii],
                     fontsize=25)

        for ii in range(1, nCols_new):
            self.ax.plot(self.xs[incr_cont[ii-1]:incr_cont[ii]],
                         self.ys[incr_cont[ii-1]:incr_cont[ii]],
                         self.color_new[ii], marker='o', label=self.seeg_labels[incr_cont[ii-1]])
            self.ax.text(self.xs[incr_cont[ii-1]],
                         self.ys[incr_cont[ii-1]],
                         str(self.seeg_labels[incr_cont[ii-1]]), color=self.color_new[ii], fontsize=35)

        for er in range(numregions):
            self.ax.text(self.xreg[er], self.yreg[er] +
                         0.7, str(er+1), color='g', fontsize=22)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        if titlestr is None:
            self.ax.set_title('SEEG Implantations for ' +
                              ' nez=' + str(ezindices.size) +
                              ' npz=' + str(pzindices.size), **self.title_font)
        else:
            self.ax.set_title(titlestr)
        self.ax.grid(True)
        self.ax.legend(loc=loc, fontsize=35)

        return self.fig, self.ax
