import os
import numpy as np
import pandas as pd
import scipy

import sys
sys.path.append('../')
import peakdetect
import zipfile

import math as m

from scipy.signal import butter, lfilter
'''
Module: Util
Description: These functions and objects are used to assist in setting up any sort of simulation environment. 

PostProcess helps analyze the simulated data and perform rejection of senseless data and to analyze the z time series and determine an onset/offset period.

MoveContacts helps analyze the simulated data's structural input data like seeg_xyz and region_centers to determine how to move a certain seeg contact and it's corresponding electrode. In addition, it can determine the region/contact with the closest point, so that can be determined as an EZ region.
'''

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y
def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass', analog=False)
    return b, a
def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def renamefiles(patient, project_dir):
    ####### Initialize files needed to 
    # convert seeg.xyz to seeg.txt file
    sensorsfile = os.path.join(project_dir, "seeg.xyz")
    newsensorsfile = os.path.join(project_dir, "seeg.txt")
    try:
        os.rename(sensorsfile, newsensorsfile)
    except:
        print("Already renamed seeg.xyz possibly!")

    # convert gain_inv-square.mat file into gain_inv-square.txt file
    gainmatfile = os.path.join(project_dir, "gain_inv-square.mat")
    newgainmatfile = os.path.join(project_dir, "gain_inv-square.txt")
    try:
        os.rename(gainmatfile, newgainmatfile)
    except:
        print("Already renamed gain_inv-square.mat possibly!")

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

def getall_sourceandelecs(confile, seegfile, project_dir):
    ####### Initialize files needed to 
    sensorsfile = os.path.join(project_dir, "seeg.txt")
    confile = os.path.join(project_dir, "connectivity.zip")

    # extract the seeg_xyz coords and the region centers
    seeg_xyz = tvbsim.util.extractseegxyz(sensorsfile)
    con = initconn(confile)

    return seeg_xyz, con

def read_surf(directory, use_subcort):
    '''
    Pass in directory for where the entire metadata for this patient is
    '''
    # Shift to account for 0 - unknown region, not included later
    reg_map_cort = np.genfromtxt((os.path.join(directory, "region_mapping_cort.txt")), dtype=int) - 1
    reg_map_subc = np.genfromtxt((os.path.join(directory, "region_mapping_subcort.txt")), dtype=int) - 1

    with zipfile.ZipFile(os.path.join(directory, "surface_cort.zip")) as zip:
        with zip.open('vertices.txt') as fhandle:
            verts_cort = np.genfromtxt(fhandle)
        with zip.open('normals.txt') as fhandle:
            normals_cort = np.genfromtxt(fhandle)
        with zip.open('triangles.txt') as fhandle:
            triangles_cort = np.genfromtxt(fhandle, dtype=int)

    with zipfile.ZipFile(os.path.join(directory, "surface_subcort.zip")) as zip:
        with zip.open('vertices.txt') as fhandle:
            verts_subc = np.genfromtxt(fhandle)
        with zip.open('normals.txt') as fhandle:
            normals_subc = np.genfromtxt(fhandle)
        with zip.open('triangles.txt') as fhandle:
            triangles_subc = np.genfromtxt(fhandle, dtype=int)

    vert_areas_cort = compute_vertex_areas(verts_cort, triangles_cort)
    vert_areas_subc = compute_vertex_areas(verts_subc, triangles_subc)

    if not use_subcort:
        return (verts_cort, normals_cort, vert_areas_cort, reg_map_cort)
    else:
        verts = np.concatenate((verts_cort, verts_subc))
        normals = np.concatenate((normals_cort, normals_subc))
        areas = np.concatenate((vert_areas_cort, vert_areas_subc))
        regmap = np.concatenate((reg_map_cort, reg_map_subc))

        return (verts, normals, areas, regmap)
def compute_triangle_areas(vertices, triangles):
    """Calculates the area of triangles making up a surface."""
    tri_u = vertices[triangles[:, 1], :] - vertices[triangles[:, 0], :]
    tri_v = vertices[triangles[:, 2], :] - vertices[triangles[:, 0], :]
    tri_norm = np.cross(tri_u, tri_v)
    triangle_areas = np.sqrt(np.sum(tri_norm ** 2, axis=1)) / 2.0
    triangle_areas = triangle_areas[:, np.newaxis]
    return triangle_areas
def compute_vertex_areas(vertices, triangles):
    triangle_areas = compute_triangle_areas(vertices, triangles)
    vertex_areas = np.zeros((vertices.shape[0]))
    for triang, vertices in enumerate(triangles):
        for i in range(3):
            vertex_areas[vertices[i]] += 1./3. * triangle_areas[triang]
    return vertex_areas
def gain_matrix_inv_square(vertices, areas, region_mapping,
                       nregions, sensors):
    '''
    Computes a gain matrix using an inverse square fall off (like a mean field model)
    Parameters
    ----------
    vertices             np.ndarray of floats of size n x 3, where n is the number of vertices
    areas                np.ndarray of floats of size n x 3
    region_mapping       np.ndarray of ints of size n
    nregions             int of the number of regions
    sensors              np.ndarray of floats of size m x 3, where m is the number of sensors

    Returns
    -------
    np.ndarray of size m x n
    '''

    nverts = vertices.shape[0]
    nsens = sensors.shape[0]

    reg_map_mtx = np.zeros((nverts, nregions), dtype=int)
    for i, region in enumerate(region_mapping):
       if region >= 0:
           reg_map_mtx[i, region] = 1

    gain_mtx_vert = np.zeros((nsens, nverts))
    for sens_ind in range(nsens):
        a = sensors[sens_ind, :] - vertices
        na = np.sqrt(np.sum(a**2, axis=1))
        gain_mtx_vert[sens_ind, :] = areas / na**2

    return gain_mtx_vert.dot(reg_map_mtx)

class PostProcess():
    '''
    '''
    def __init__(self, epits, seegts, times):
        self.epits = epits
        self.seegts = seegts
        self.times = times

    def postprocts(self, samplerate=1000):
        # reject certain 5 seconds of simulation
        secstoreject = 15
        sampstoreject = secstoreject * samplerate

        # get the time series processed and squeezed that we want to save
        new_times = self.times[sampstoreject:]
        new_epits = self.epits[sampstoreject:, 1, :, :].squeeze().T
        new_zts = self.epits[sampstoreject:, 0, :, :].squeeze().T
        new_seegts = self.seegts[sampstoreject:, :, :, :].squeeze().T

        # don't reject any time period
        # new_times = self.times
        # new_epits = self.epits[:, 1, :, :].squeeze().T
        # new_zts = self.epits[:, 0, :, :].squeeze().T
        # new_seegts = self.seegts[:,:, :, :].squeeze().T

        return new_times, new_epits, new_seegts, new_zts

    # assuming onset is the first bifurcation and then every other one is onsets
    # every other bifurcation after the first one is the offset
    def findonsetoffset(self, zts, delta=0.2/8):
        maxpeaks, minpeaks = peakdetect.peakdetect(zts.squeeze(), delta=delta)
        minpeaks = np.asarray(minpeaks)
        maxpeaks = np.asarray(maxpeaks)

        # get every other peaks
        if minpeaks.ndim == 2:
            onsettime = minpeaks[:,0]
        else:
            # print("minpeaks are: ", minpeaks)
            onsettime = np.array([np.nan])
        # get every other peaks
        if maxpeaks.ndim == 2:
            offsettime = maxpeaks[:,0]
        else:
            # print("maxpeaks are ", maxpeaks)
            offsettime = np.array([np.nan])

        # get the difference in size of onset/offset arrays
        diffsize = abs(onsettime.size - offsettime.size)
        # pad the arrays to have nans if the array sizes are uneven
        if onsettime.size > offsettime.size:
            offsettime = np.append(offsettime, np.nan*np.ones(diffsize), axis=0)
        elif onsettime.size < offsettime.size:
            onsettime = np.append(offsettime, np.nan*np.ones(diffsize), axis=0)
        
        return onsettime, offsettime

    def getseiztimes(self, settimes):
        # perform some checks
        if settimes.size == 0:
            print("no onset/offset available!")
            return 0

        # sort in place the settimes by onsets, since those will forsure have 1
        settimes = settimes[settimes[:,0].argsort()]

        # get the onsets/offset pairs now
        onsettimes = settimes[:,0]
        offsettimes = settimes[:,1]
        seizonsets = []
        seizoffsets = []
        
        # start loop after the first onset/offset pair
        for i in range(0,len(onsettimes)):        
            # get current onset/offset times
            curronset = onsettimes[i]
            curroffset = offsettimes[i]

            # handle first case
            if i == 0:
                prevonset = curronset
                prevoffset = curroffset
                seizonsets.append(prevonset)
            # check current onset/offset
            else:
                # if the onset now is greater then the offset
                # we have one seizure instance
                if curronset > prevoffset:
                    seizoffsets.append(prevoffset)
                    prevonset = curronset
                    prevoffset = curroffset
                    seizonsets.append(prevonset)
                else:
                    # just move the offset along
                    prevoffset = curroffset
            # if at any point, offset is nan, then just return
            if np.isnan(prevoffset):
                print('returning cuz prevoffset is nan!')
                return seizonsets, seizoffsets

        if not np.isnan(prevoffset):
            seizoffsets.append(prevoffset)

        return seizonsets, seizoffsets

    def getonsetsoffsets(self, zts, indices,delta=0.2/8):
        # create lambda function for checking the indices
        check = lambda indices: isinstance(indices,np.ndarray) and len(indices)>=1

        onsettimes = []
        offsettimes = []
        settimes = []

        # go through and get onset/offset times of ez indices
        if check(indices):
            for index in indices:
                _onsettimes, _offsettimes = self.findonsetoffset(zts[index, :].squeeze(), delta=delta)
                settimes.append(list(zip(_onsettimes, _offsettimes)))
                
        # flatten out list structure if there is one
        settimes = [item for sublist in settimes for item in sublist]
        settimes = np.asarray(settimes).squeeze()

        # do an error check and reshape arrays if necessary
        if settimes.ndim == 1:
            settimes = settimes.reshape(1,settimes.shape[0])

        return settimes

class MoveContacts():
    '''
    An object wrapper for all the functionality in moving a contact during TVB
    simulation.

    Will be able to move contacts, compute a new xyz coordinate map of the contacts and
    re-compute a gain matrix.
    '''
    def __init__(self, seeg_labels=[], seeg_xyz=[], region_labels=[], reg_xyz=[], VERBOSE=False):
        self.seeg_xyz = seeg_xyz
        self.reg_xyz = reg_xyz

        self.seeg_labels = seeg_labels
        self.region_labels = region_labels

        if type(self.seeg_xyz) is not np.ndarray and len(seeg_xyz) > 0:
            self.seeg_xyz = pd.DataFrame.as_matrix(self.seeg_xyz)
        if type(self.reg_xyz) is not np.ndarray and len(reg_xyz) > 0:
            self.reg_xyz = pd.DataFrame.as_matrix(self.reg_xyz)
                
        self.VERBOSE=VERBOSE

    ''' 
    Functions for loading data into the object
    '''
    def set_seegxyz(self, seeg_xyz):
        self.seeg_xyz = seeg_xyz
    def load_regionlabels(self, region_labels):
        self.region_labels = region_labels
    def load_regxyz(self, reg_xyz):
        self.reg_xyz = reg_xyz
    def load_seeglabels(self, seeg_labels):
        self.seeg_labels = seeg_labels

    def simplest_gain_matrix(self, seeg_xyz):
        '''
        This is a function to recompute a new gain matrix based on xyz that moved
        G = 1 / ( 4*pi * sum(sqrt(( X - X[:, new])^2))^2)
        '''
        # NOTE IF YOU MOVE SEEGXYZ ONTO REGXYZ, YOU DIVIDE BY 0, SO THERE IS A PROBLEM
        #reg_xyz = con.centres
        dr = self.reg_xyz - seeg_xyz[:, np.newaxis]
        if 0 in dr:
            print("Computing simplest gain matrix will result \
                in error when contact is directly on top of any region!\
                Dividing by 0!")

        ndr = np.sqrt((dr**2).sum(axis=-1))
        Vr = 1.0 / (4 * np.pi) / ndr**2
        return Vr

    def getallcontacts(self, seeg_contact):
        '''
        Gets the entire electrode contacts' indices, so that we can modify the corresponding xyz
        '''
        # get the elec label name
        isleftside = seeg_contact.find("'")
        if self.VERBOSE:
            print(seeg_contact)
        
        contacts = []
        for tempcontact in self.seeg_labels:
            for idx, s in enumerate(tempcontact):
                if s.isdigit():
                    elec_label = tempcontact[0:idx]
                    break
            contacts.append((elec_label, int(tempcontact[len(elec_label):])))

        # get indices depending on if it is a left/right hemisphere electrode
        if isleftside != -1:
            elec_label = seeg_contact.split("'")[0]
            electrodeindices = [i for i,item in enumerate(self.seeg_labels) if elec_label+"'" in item]
        else:
            for idx, s in enumerate(seeg_contact):
                if s.isdigit():
                    elec_label = seeg_contact[0:idx]
                    break
            electrodeindices = [i for i,item in enumerate(contacts) if elec_label == item[0]]
        print('\nelec label is %s' % elec_label)
        return electrodeindices

    def getindexofregion(self, region):
        '''
        This is a helper function to determine the indices of the ez and pz region
        '''
        sorter = np.argsort(self.region_labels)
        indice = sorter[np.searchsorted(self.region_labels, region, sorter=sorter)]
        return indice

    def findclosestcontact(self, ezindex, elecmovedindices=[]):
        '''
        This function finds the closest contact to an ezregion
        '''
        # get the ez region's xyz coords
        ez_regionxyz = self.reg_xyz[ezindex]

        # create a mask of the indices we already moved
        elec_indices = np.arange(0, self.seeg_xyz.shape[0])
        movedmask = [element for i, element in enumerate(elec_indices) if i not in elecmovedindices]

        # create a spatial KD tree -> find closest SEEG contact to region in Euclidean
        tree = scipy.spatial.KDTree(self.seeg_xyz[movedmask, :])
        near_seeg = tree.query(ez_regionxyz)
        
        # get the distance and the index at the min
        distance = near_seeg[0]
        seeg_index = near_seeg[1]
        return seeg_index, distance

    def movecontact(self, ezindex, seeg_index):
        '''
        This function moves the contact and the entire electrode the correct distance, so that the contact
        is on the ezregion now
        '''
        ez_regionxyz = self.reg_xyz[ezindex,:]
        closest_seeg = self.seeg_xyz[seeg_index,:].copy()
        seeg_contact = self.seeg_labels[seeg_index]

        seeg_label = seeg_contact.split("'")[0]
        # perform some processing to get all the contact indices for this electrode
        electrodeindices = self.getallcontacts(seeg_contact)

        print(closest_seeg)

        # get the euclidean distance that will be moved for this electrode
        x_dist = ez_regionxyz[0] - closest_seeg[0]
        y_dist = ez_regionxyz[1] - closest_seeg[1]
        z_dist = ez_regionxyz[2] - closest_seeg[2]
        distancetomove = [x_dist, y_dist, z_dist]

        # createa copy of the seeg_xyz df and modify the electrode
        new_seeg_xyz = self.seeg_xyz.copy()
        new_seeg_xyz[electrodeindices] = new_seeg_xyz[electrodeindices] + distancetomove

        # modify the object's seeg xyz
        self.seeg_xyz[electrodeindices] = self.seeg_xyz[electrodeindices] + distancetomove

        # print(new_seeg_xyz-ez_regionxyz)

        if self.VERBOSE:
            print("\n\n movecontact function summary: \n")
            print("Closest contact to ezregion: ", ez_regionxyz, ' is ', seeg_contact)
            print("That is located at: ", closest_seeg)
            print("It will move: ", distancetomove)
            print("New location after movement is", new_seeg_xyz[seeg_index])
            # print electrodeindices
        
        return new_seeg_xyz, electrodeindices

    def cart2sph(self, x, y, z):
        '''
        Transform Cartesian coordinates to spherical
        
        Paramters:
        x           (float) X coordinate
        y           (float) Y coordinate
        z           (float) Z coordinate

        :return: radius, elevation, azimuth
        '''
        x2_y2 = x**2 + y**2
        r = m.sqrt(x2_y2 + z**2)                    # r
        elev = m.atan2(m.sqrt(x2_y2), z)            # Elevation / phi
        az = m.atan2(y, x)                          # Azimuth / theta
        return r, elev, az

    def sph2cart(self, r, elev, az):
        x = r*m.sin(elev)*m.cos(az)
        y = r*m.sin(elev)*m.sin(az)
        z = r*m.cos(elev)
        return x,y,z


    def movecontactto(self, ezindex, seeg_index, distance=0, axis='auto'):
        '''
        This function moves the contact and the entire electrode the correct distance, so that the contact
        is on the ezregion now
        '''
        ez_regionxyz = self.reg_xyz[ezindex,:] # get xyz of ez region
        closest_seeg = self.seeg_xyz[seeg_index,:].copy() # get the closest seeg's xyz
        seeg_contact = self.seeg_labels[seeg_index] # get the closest seeg's label

        seeg_label = seeg_contact.split("'")[0]
        # perform some processing to get all the contact indices for this electrode
        electrodeindices = self.getallcontacts(seeg_contact)

        # dist = np.sqrt(distance**2 / 3.)
        # get the euclidean distance that will be moved for this electrode
        x_dist = ez_regionxyz[0] - closest_seeg[0]
        y_dist = ez_regionxyz[1] - closest_seeg[1]
        z_dist = ez_regionxyz[2] - closest_seeg[2]

        distancetomove = [x_dist, y_dist, z_dist]

        if axis == 'auto' and distance > 0:
            # modify the distance in sphereical coordinates
            r, elev, az = self.cart2sph(x_dist, y_dist, z_dist)
            r = r - distance 
            x_dist, y_dist, z_dist = self.sph2cart(r, elev, az)

            distancetomove = [x_dist, y_dist, z_dist]
        elif axis=='x':
            # move x
            pass
        elif axis=='y':
            # move y
            pass
        elif axis=='z':
            # move z
            pass

        # createa copy of the seeg_xyz df and modify the electrode
        new_seeg_xyz = self.seeg_xyz.copy()
        new_seeg_xyz[electrodeindices] = new_seeg_xyz[electrodeindices] + distancetomove

        # modify the object's seeg xyz
        self.seeg_xyz[electrodeindices] = self.seeg_xyz[electrodeindices] + distancetomove

        if self.VERBOSE:
            print("\n\n movecontact function summary: \n")
            print("Closest contact to ezregion: ", ez_regionxyz, ' is ', seeg_contact)
            print("That is located at: ", closest_seeg)
            print("It will move: ", distancetomove)
            print("New location after movement is", new_seeg_xyz[seeg_index], '\n')
            # print electrodeindices
        
        return new_seeg_xyz, electrodeindices

    def findclosestregion(self, seegindex, p=2):
        '''
        This function finds the closest contact to an ezregion
        '''
        # get the ez region's xyz coords
        contact_xyz = self.seeg_xyz[seegindex]

        # create a spatial KD tree -> find closest SEEG contact to region in Euclidean
        tree = scipy.spatial.KDTree(self.reg_xyz)
        near_region = tree.query(contact_xyz, p=p)
        
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

    # print gainmat.shape
    # print seeg_contact
    # print seeg_xyz.iloc[electrodeindices]
    # print new_seeg_xyz.iloc[electrodeindices]

    # # print near_seeg[1].ravel()
    # print seeg_xyz.iloc[near_seeg[1]]