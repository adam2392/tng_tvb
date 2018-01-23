import os
import numpy as np
import pandas as pd
import scipy

import sys
sys.path.append('../')
import peakdetect
import zipfile

'''
Module: Util
Description: These functions and objects are used to assist in setting up any sort of simulation environment. 

PostProcess helps analyze the simulated data and perform rejection of senseless data and to analyze the z time series and determine an onset/offset period.

MoveContacts helps analyze the simulated data's structural input data like seeg_xyz and region_centers to determine how to move a certain seeg contact and it's corresponding electrode. In addition, it can determine the region/contact with the closest point, so that can be determined as an EZ region.
'''

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
        # secstoreject = 7
        # sampstoreject = secstoreject * samplerate

        # # get the time series processed and squeezed that we want to save
        # new_times = self.times[sampstoreject:]
        # new_epits = self.epits[sampstoreject:, 1, :, :].squeeze().T
        # new_zts = self.epits[sampstoreject:, 0, :, :].squeeze().T
        # new_seegts = self.seegts[sampstoreject:, :, :, :].squeeze().T

        # don't reject any time period
        new_times = self.times
        new_epits = self.epits[:, 1, :, :].squeeze().T
        new_zts = self.epits[:, 0, :, :].squeeze().T
        new_seegts = self.seegts[:,:, :, :].squeeze().T

        return new_times, new_epits, new_seegts, new_zts

    # assuming onset is the first bifurcation and then every other one is onsets
    # every other bifurcation after the first one is the offset
    def findonsetoffset(self, zts, delta=0.2/8):
        maxpeaks, minpeaks = peakdetect.peakdetect(zts, delta=delta)
        
        # get every other peaks
        onsettime, _ = zip(*minpeaks)
        offsettime, _ = zip(*maxpeaks)
        
        return onsettime, offsettime

    def getseiztimes(self, onsettimes, offsettimes):
        minsize = np.min((len(onsettimes),len(offsettimes)))
        seizonsets = []
        seizoffsets = []
        
        # perform some checks
        if minsize == 0:
            print("no full onset/offset available!")
            return 0
        
        idx = 0
        # to store the ones we are checking rn
        _onset = onsettimes[idx]
        _offset = offsettimes[idx]
        seizonsets.append(_onset)
        
        # start loop after the first onset/offset pair
        for i in range(1,minsize):        
            # to store the previoius values
            _nextonset = onsettimes[i]
            _nextoffset = offsettimes[i]
            
            # check this range and add the offset if it was a full seizure
            # before the next seizure
            if _nextonset < _offset:
                _offset = _nextoffset
            else:
                seizoffsets.append(_offset)
                idx = i
                # to store the ones we are checking rn
                _onset = onsettimes[idx]
                _offset = offsettimes[idx]
                seizonsets.append(_onset)
        if len(seizonsets) != len(seizoffsets):
            seizonsets = seizonsets[0:len(seizoffsets)]
        return seizonsets, seizoffsets
            
    def getonsetsoffsets(self, zts, ezindices, pzindices):
        # create lambda function for checking the indices
        check = lambda indices: isinstance(indices,np.ndarray) and len(indices)>=1

        onsettimes=np.array([])
        offsettimes=np.array([])
        if check(ezindices):
            for ezindex in ezindices:
                _onsettimes, _offsettimes = postprocessor.findonsetoffset(zts[ezindex, :].squeeze(), 
                                                                        delta=0.2/8)
                onsettimes = np.append(onsettimes, np.asarray(_onsettimes))
                offsettimes = np.append(offsettimes, np.asarray(_offsettimes))

        if check(pzindices):
            for pzindex in pzindices:
                _onsettimes, _offsettimes = postprocessor.findonsetoffset(zts[pzindex, :].squeeze(), 
                                                                        delta=0.2/8)
                onsettimes = np.append(onsettimes, np.asarray(_onsettimes))
                offsettimes = np.append(offsettimes, np.asarray(_offsettimes))

        # first sort onsettimes and offsettimes
        onsettimes.sort()
        offsettimes.sort()
        
        return onsettimes, offsettimes

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

    def set_seegxyz(self, seeg_xyz):
        self.seeg_xyz = seeg_xyz

    def simplest_gain_matrix(self, seeg_xyz):
        '''
        This is a function to recompute a new gain matrix based on xyz that moved
        G = 1 / ( 4*pi * sum(sqrt(( X - X[:, new])^2))^2)
        '''
        # NOTE IF YOU MOVE SEEGXYZ ONTO REGXYZ, YOU DIVIDE BY 0, SO THERE IS A PROBLEM
        #reg_xyz = con.centres
        dr = self.reg_xyz - seeg_xyz[:, np.newaxis]

        if 0 in dr:
            print("Computing simplest gain matrix will result in error! Dividing by 0!")

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
            print(seeg_contact)
            print(elec_label)
        
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
        ez_regionxyz = self.reg_xyz[ezindex]
        closest_seeg = self.seeg_xyz[seeg_index]
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

    def movecontactto(self, ezindex, seeg_index, distance=0, axis='auto'):
        '''
        This function moves the contact and the entire electrode the correct distance, so that the contact
        is on the ezregion now
        '''
        ez_regionxyz = self.reg_xyz[ezindex] # get xyz of ez region
        closest_seeg = self.seeg_xyz[seeg_index] # get the closest seeg's xyz
        seeg_contact = self.seeg_labels[seeg_index] # get the closest seeg's label

        seeg_label = seeg_contact.split("'")[0]
        # perform some processing to get all the contact indices for this electrode
        electrodeindices = self.getallcontacts(seeg_contact)

        # get the euclidean distance that will be moved for this electrode
        x_dist = ez_regionxyz[0] - closest_seeg[0]
        y_dist = ez_regionxyz[1] - closest_seeg[1]
        z_dist = ez_regionxyz[2] - closest_seeg[2]
        distancetomove = [x_dist, y_dist, z_dist]

        if axis == 'auto' and distance != 0:
            # note: the current method moves the contact in the direction of the original
            # contact's position before movement
            # move all 3, just perturb, so |distancetomove| - perturb == distance
            dist = np.sqrt(distance**2 / 3.)
            x_dist = min(abs(x_dist-dist), abs(x_dist+dist))
            y_dist = min(abs(y_dist-dist), abs(y_dist+dist))
            z_dist = min(abs(z_dist-dist), abs(z_dist+dist))
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
            print("New location after movement is", new_seeg_xyz[seeg_index])
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