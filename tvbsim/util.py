import sys
sys.path.append('../')
from tvb.simulator.lab import *
import os.path

import numpy as np
import pandas as pd
import scipy

def simplest_gain_matrix(seeg_xyz, reg_xyz=np.ndarray):
    '''
    This is a function to recompute a new gain matrix based on xyz that moved
    G = 1 / ( 4*pi * sum(sqrt(( X - X[:, new])^2))^2)
    '''
    #reg_xyz = con.centres
    dr = reg_xyz - seeg_xyz[:, np.newaxis]
    ndr = np.sqrt((dr**2).sum(axis=-1))
    Vr = 1.0 / (4 * np.pi) / ndr**2
    return Vr

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

def findclosestcontact(ezindex, region_centers, seeg_xyz):
    '''
    This function finds the closest contact to the ezregion
    '''
    # get the ez region's xyz coords
    ez_regionxyz = region_centers[ezindex]

    # convert seeg_xyz dataframe to np array
    if type(seeg_xyz) is not np.ndarray:
        seeg_xyz = pd.DataFrame.as_matrix(seeg_xyz)
    
    # create a spatial KD tree -> find closest SEEG contact to region in Euclidean
    tree = scipy.spatial.KDTree(seeg_xyz)
    near_seeg = tree.query(ez_regionxyz)
    distance = near_seeg[0]
    seeg_index = near_seeg[1]

    return seeg_index, distance

def movecontact(seeg_xyz, region_centers, ezindex, seeg_index):
    '''
    This function moves the contact and the entire electrode the correct distance, so that the contact
    is on the ezregion now
    '''
    ez_regionxyz = region_centers[ezindex]
    closest_seeg = seeg_xyz.iloc[seeg_index]

    # perform some processing to get all the contact indices for this electrode
    seeg_labels = np.array(seeg_xyz.index, dtype='str')
    seeg_contact = seeg_xyz.iloc[seeg_index].index[0]
    electrodeindices = getallcontacts(seeg_labels, seeg_contact)

    # get the euclidean distance that will be moved for this electrode
    x_dist = closest_seeg['x'].values[0] - ez_regionxyz[0][0]
    y_dist = closest_seeg['y'].values[0] - ez_regionxyz[0][1]
    z_dist = closest_seeg['z'].values[0] - ez_regionxyz[0][2]
    distancetomove = [x_dist, y_dist, z_dist]

    # createa copy of the seeg_xyz df and modify the electrode
    new_seeg_xyz = seeg_xyz.copy()
    new_seeg_xyz.iloc[electrodeindices] = new_seeg_xyz.iloc[electrodeindices] - distancetomove

    print "Closest contact to ezregion: ", region_centers[ezindex], ' is ', seeg_contact
    print "That is located at: ", closest_seeg
    print "It will move: ", distancetomove
    print "New location after movement is", new_seeg_xyz.iloc[seeg_index]

    return new_seeg_xyz

def getallcontacts(seeg_labels, seeg_contact):
    '''
    Gets the entire electrode contacts' indices, so that we can modify the corresponding xyz
    '''
    # get the elec label name
    elec_label = seeg_contact.split("'")[0]
    isleftside = seeg_contact.find("'")
    
    # get indices depending on if it is a left/right hemisphere electrode
    if isleftside != -1:
        print 'is left'
        electrodeindices = [i for i,item in enumerate(seeg_labels) if elec_label+"'" in item]
    else:
        electrodeindices = [i for i,item in enumerate(seeg_labels) if elec_label in item]
    return electrodeindices
        
def getindexofregion(regions, ezregion, pzregion=[]):
    '''
    This is a helper function to determine the indices of the ez and pz regions
    '''
    sorter = np.argsort(regions)
    ezindices = sorter[np.searchsorted(regions, ezregion, sorter=sorter)]
    pzindices = sorter[np.searchsorted(regions, pzregion, sorter=sorter)]

    return ezindices, pzindices

def getregionsforcontacts(regions, region_centers, seeg_point):
    # seeg_point is xyz
    x, y, z = seeg_point.ravel()
    
    # return region that is closest to seeg_point
    pass

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