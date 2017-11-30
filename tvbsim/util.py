import os
import numpy as np
import pandas as pd
import scipy

def extractseegxyz(seegfile):
    '''
    This is just a wrapper function to retrieve the seeg coordinate data in a pd dataframe
    '''
    seeg_pd = pd.read_csv(seegfile, names=['x','y','z'], delim_whitespace=True)
    return seeg_pd
def extractcon(confile):
    '''
    This is a wrapper function to obtain the connectivity object from a file 
    '''
    con = connectivity.Connectivity.from_file(confile)
    return con

class PostProcess():
    '''
    '''
    def __init__(self, epits, seegts, times):
        self.epits = epits
        self.seegts = seegts
        self.times = times

    def postprocts(self, samplerate=1000):
        # reject certain 5 seconds of simulation
        secstoreject = 7
        sampstoreject = secstoreject * samplerate

        # get the time series processed and squeezed that we want to save
        new_times = self.times[sampstoreject:]
        new_epits = self.epits[sampstoreject:, 1, :, :].squeeze().T
        new_zts = self.epits[sampstoreject:, 0, :, :].squeeze().T
        new_seegts = self.seegts[sampstoreject:, :, :, :].squeeze().T

        # don't reject any time period
        new_times = self.times
        new_epits = self.epits[:, 1, :, :].squeeze().T
        new_zts = self.epits[:, 0, :, :].squeeze().T
        new_seegts = self.seegts[:,:, :, :].squeeze().T

        return new_times, new_epits, new_seegts, new_zts

    # assuming onset is the first bifurcation and then every other one is onsets
    # every other bifurcation after the first one is the offset
    def findonsetoffset(self, zts):
        maxpeaks, minpeaks = peakdetect.peakdetect(zts, delta=0.2)
        
        # get every other peaks
        onsettime, _ = zip(*maxpeaks)
        offsettime, _ = zip(*minpeaks)
        
        return onsettime, offsettime

class MoveContacts():
    '''
    An object wrapper for all the functionality in moving a contact during TVB
    simulation.

    Will be able to move contacts, compute a new xyz coordinate map of the contacts and
    re-compute a gain matrix.
    '''
    def __init__(self, seeg_labels, seeg_xyz, region_labels, reg_xyz, VERBOSE=False):
        self.seeg_xyz = seeg_xyz
        self.reg_xyz = reg_xyz

        self.seeg_labels = seeg_labels
        self.region_labels = region_labels

        if type(self.seeg_xyz) is not np.ndarray:
            self.seeg_xyz = pd.DataFrame.as_matrix(self.seeg_xyz)
        if type(self.reg_xyz) is not np.ndarray:
            self.reg_xyz = pd.DataFrame.as_matrix(self.reg_xyz)
                
        self.VERBOSE=VERBOSE

    def simplest_gain_matrix(self, seeg_xyz):
        '''
        This is a function to recompute a new gain matrix based on xyz that moved
        G = 1 / ( 4*pi * sum(sqrt(( X - X[:, new])^2))^2)
        '''
        #reg_xyz = con.centres
        dr = self.reg_xyz - seeg_xyz[:, np.newaxis]
        ndr = np.sqrt((dr**2).sum(axis=-1))
        Vr = 1.0 / (4 * np.pi) / ndr**2
        return Vr

    def getallcontacts(self, seeg_contact):
        '''
        Gets the entire electrode contacts' indices, so that we can modify the corresponding xyz
        '''
        # get the elec label name
        elec_label = seeg_contact.split("'")[0]
        isleftside = seeg_contact.find("'")
        if self.VERBOSE:
            print seeg_contact
            print elec_label
        
        # get indices depending on if it is a left/right hemisphere electrode
        if isleftside != -1:
            electrodeindices = [i for i,item in enumerate(self.seeg_labels) if elec_label+"'" in item]
        else:
            electrodeindices = [i for i,item in enumerate(self.seeg_labels) if elec_label in item]
        return electrodeindices

    def getindexofregion(self, region):
        '''
        This is a helper function to determine the indices of the ez and pz region
        '''
        sorter = np.argsort(self.region_labels)
        indice = sorter[np.searchsorted(self.region_labels, region, sorter=sorter)]
        return indice

    def findclosestcontact(self, ezindex):
        '''
        This function finds the closest contact to an ezregion
        '''
        # get the ez region's xyz coords
        ez_regionxyz = self.reg_xyz[ezindex]

        # create a spatial KD tree -> find closest SEEG contact to region in Euclidean
        tree = scipy.spatial.KDTree(self.seeg_xyz)
        near_seeg = tree.query(ez_regionxyz)
        
        distance = near_seeg[0]
        seeg_index = near_seeg[1]
        return seeg_index, distance

    def movecontact(self, ezindex, seeg_index):
        '''
        This function moves the contact and the entire electrode the correct distance, so that the contact
        is on the ezregion now
        '''
        ez_regionxyz = self.reg_xyz[ezindex]
        closest_seeg = self.seeg_xyz[seeg_index]
        seeg_contact = self.seeg_labels[seeg_index]

        # perform some processing to get all the contact indices for this electrode
        electrodeindices = self.getallcontacts(seeg_contact)

        print closest_seeg

        # get the euclidean distance that will be moved for this electrode
        x_dist = ez_regionxyz[0] - closest_seeg[0]
        y_dist = ez_regionxyz[1] - closest_seeg[1]
        z_dist = ez_regionxyz[2] - closest_seeg[2]
        distancetomove = [x_dist, y_dist, z_dist]

        # createa copy of the seeg_xyz df and modify the electrode
        new_seeg_xyz = self.seeg_xyz.copy()
        new_seeg_xyz[electrodeindices] = new_seeg_xyz[electrodeindices] + distancetomove

        if self.VERBOSE:
            print "\n\n movecontact function summary: \n"
            print "Closest contact to ezregion: ", ez_regionxyz, ' is ', seeg_contact
            print "That is located at: ", closest_seeg
            print "It will move: ", distancetomove
            print "New location after movement is", new_seeg_xyz[seeg_index]
            # print electrodeindices
            print "\n\n"
        
        return new_seeg_xyz

    def findclosestregion(self, seegindex):
        '''
        This function finds the closest contact to an ezregion
        '''
        # get the ez region's xyz coords
        contact_xyz = self.seeg_xyz[seegindex]

        # create a spatial KD tree -> find closest SEEG contact to region in Euclidean
        tree = scipy.spatial.KDTree(self.reg_xyz)
        near_region = tree.query(contact_xyz)
        
        distance = near_region[0]
        region_index = near_region[1]
        return region_index, distance

    def getregionsforcontacts(self, seeg_contact):
        contact_index = np.where(self.seeg_labels == seeg_contact)[0]
        
        # determine the region index and distance to closest region
        region_index, distance = self.findclosestregion(contact_index)
        
        return region_index, distance


if __name__ == '__main__':
    patient = 'id001_ac'
    project_dir = '/Users/adam2392/Documents/tvb/metadata/'
    confile = os.path.join(project_dir, patient, "connectivity.zip")
    ####################### 1. Extract Relevant Info ########################
    con = extractcon(confile)
    region_centers = con.centres
    regions = con.region_labels
    seegfile = os.path.join(project_dir, patient, "seeg.txt")
    seeg_xyz = extractseegxyz(seegfile)

    # first get all contacts of the same electrode
    seeg_labels = np.array(seeg_xyz.index, dtype='str')

    # determine closest contact for region
    ezregion = ['ctx-lh-bankssts']
    ezindice, pzindice = getindexofregion(regions, ezregion)
    near_seeg = findclosestcontact(ezindice, region_centers, seeg_xyz)

    # now move contact and recompute gain matrix
    seeg_contact = np.array(seeg_xyz.iloc[near_seeg[1]].index, dtype='str')[0]
    electrodeindices = getallcontacts(seeg_labels, seeg_contact)

    new_seeg_xyz = movecontact(seeg_xyz, region_centers, ezindice, seeg_contact)

    gainmat = simplest_gain_matrix(new_seeg_xyz.as_matrix(), reg_xyz=region_centers)

    print gainmat.shape
    print seeg_contact
    print seeg_xyz.iloc[electrodeindices]
    print new_seeg_xyz.iloc[electrodeindices]

    # print near_seeg[1].ravel()
    print seeg_xyz.iloc[near_seeg[1]]