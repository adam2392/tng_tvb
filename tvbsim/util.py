import os
import numpy as np
import pandas as pd
import scipy

import sys
sys.path.append('../')
import zipfile
import math as m
from shutil import copyfile

'''
Module: Util
Description: These functions and objects are used to assist in setting up any sort of simulation environment. 

PostProcess helps analyze the simulated data and perform rejection of senseless data and to analyze the z time series and determine an onset/offset period.

MoveContacts helps analyze the simulated data's structural input data like seeg_xyz and region_centers to determine how to move a certain seeg contact and it's corresponding electrode. In addition, it can determine the region/contact with the closest point, so that can be determined as an EZ region.
'''


def renamefiles(project_dir):
    # Initialize files needed to
    # convert seeg.xyz to seeg.txt file
    sensorsfile = os.path.join(project_dir, "seeg.xyz")
    newsensorsfile = os.path.join(project_dir, "seeg.txt")
    try:
        # copyfile(sensorsfile, newsensorsfile)
        os.rename(sensorsfile, newsensorsfile)
    except:
        print("Already renamed seeg.xyz possibly!")

    # convert gain_inv-square.mat file into gain_inv-square.txt file
    gainmatfile = os.path.join(project_dir, "gain_inv-square.mat")
    newgainmatfile = os.path.join(project_dir, "gain_inv-square.txt")
    try:
        # copyfile(gainmatfile, newgainmatfile)
        os.rename(gainmatfile, newgainmatfile)
    except:
        print("Already renamed gain_inv-square.mat possibly!")


def extractseegxyz(seegfile):
    '''
    This is just a wrapper function to retrieve the seeg coordinate data in a pd dataframe
    '''
    seeg_pd = pd.read_csv(
        seegfile, names=['x', 'y', 'z'], delim_whitespace=True)
    return seeg_pd


def extractcon(confile):
    '''
    This is a wrapper function to obtain the connectivity object from a file 
    '''
    con = connectivity.Connectivity.from_file(confile)
    return con


def read_surf(directory, use_subcort):
    '''
    Pass in directory for where the entire metadata for this patient is
    '''
    # Shift to account for 0 - unknown region, not included later
    reg_map_cort = np.genfromtxt(
        (os.path.join(directory, "region_mapping_cort.txt")), dtype=int) - 1
    reg_map_subc = np.genfromtxt(
        (os.path.join(directory, "region_mapping_subcort.txt")), dtype=int) - 1

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


def findclosestregion(seegindex, p=2):
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


def getregionsforcontacts(seeg_contact):
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

    new_seeg_xyz = movecontact(
        seeg_xyz, region_centers, ezindice, seeg_contact)

    gainmat = simplest_gain_matrix(
        new_seeg_xyz.as_matrix(), reg_xyz=region_centers)

    # print gainmat.shape
    # print seeg_contact
    # print seeg_xyz.iloc[electrodeindices]
    # print new_seeg_xyz.iloc[electrodeindices]

    # # print near_seeg[1].ravel()
    # print seeg_xyz.iloc[near_seeg[1]]
