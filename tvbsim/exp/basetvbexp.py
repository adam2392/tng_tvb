import sys
sys.path.append('../_tvblibrary/')
sys.path.append('../_tvbdata/')
sys.path.append('../')
from tvb.simulator.lab import *

import zipfile
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

class TVBExp(object):
    def __init__(self, conn, condspeed=np.inf):
        self.conn = conn
        self.conn.speed = condspeed
        self.conn.cortical[:] = True
        self.conn.weights = conn.weights/np.max(conn.weights)
        self.init_cond = None

    def getepileptorparams(self):
        params = {
            'r': self.epileptors.r,
            'ks': self.epileptors.Ks,
            'tau': self.epileptors.tau,
            'tt': self.epileptors.tt,
            'x0': self.epileptors.x0,
            'nsig': self.integrator.noise.nsig,
            'ntau': self.integrator.noise.ntau,
        }
        return params
    def getconnparams(self):
        params = {
            'speed': self.conn.speed,
            'regions': self.conn.region_labels,
            'region_xyz': self.conn.centres,
        }
        return params
    def loadseegxyz(self,seegfile):
        '''
        This is just a wrapper function to retrieve the seeg coordinate data in a pd dataframe
        '''
        seeg_pd = pd.read_csv(seegfile, names=['x','y','z'], delim_whitespace=True)
        self.seegfile = seegfile
        self.seeg_labels = seeg_pd.index.values
        self.seeg_xyz = seeg_pd.as_matrix(columns=None)

    def loadgainmat(self,gainfile):
        # function to get model in its equilibrium value
        gain_pd = pd.read_csv(gainfile, header=None, delim_whitespace=True)
        self.gainfile = gainfile
        self.gainmat = gain_pd.as_matrix()

    def loadsurfdata(self,directory,use_subcort=False):
        '''
        Pass in directory for where the entire metadata for this patient is
        '''
        # Shift to account for 0 - unknown region, not included later
        reg_map_cort = np.genfromtxt((os.path.join(directory, "region_mapping_cort.txt")), dtype=int) - 1
        with zipfile.ZipFile(os.path.join(directory, "surface_cort.zip")) as zip:
            with zip.open('vertices.txt') as fhandle:
                verts_cort = np.genfromtxt(fhandle)
            with zip.open('normals.txt') as fhandle:
                normals_cort = np.genfromtxt(fhandle)
            with zip.open('triangles.txt') as fhandle:
                triangles_cort = np.genfromtxt(fhandle, dtype=int)
        vert_areas_cort = self._compute_vertex_areas(verts_cort, triangles_cort)

        if use_subcort == False:
            print('NOT USING SUBCORT')
            self.vertices = verts_cort
            self.normals = normals_cort
            self.areas = vert_areas_cort
            self.regmap = reg_map_cort
            return (verts_cort, normals_cort, vert_areas_cort, reg_map_cort)
        else:
            reg_map_subc = np.genfromtxt((os.path.join(directory, "region_mapping_subcort.txt")), dtype=int) - 1
            with zipfile.ZipFile(os.path.join(directory, "surface_subcort.zip")) as zip:
                with zip.open('vertices.txt') as fhandle:
                    verts_subc = np.genfromtxt(fhandle)
                with zip.open('normals.txt') as fhandle:
                    normals_subc = np.genfromtxt(fhandle)
                with zip.open('triangles.txt') as fhandle:
                    triangles_subc = np.genfromtxt(fhandle, dtype=int)
            vert_areas_subc = self._compute_vertex_areas(verts_subc, triangles_subc)

            verts = np.concatenate((verts_cort, verts_subc))
            normals = np.concatenate((normals_cort, normals_subc))
            areas = np.concatenate((vert_areas_cort, vert_areas_subc))
            regmap = np.concatenate((reg_map_cort, reg_map_subc))
            self.vertices = verts
            self.normals = normals
            self.areas = areas
            self.regmap = regmap
            return (verts, normals, areas, regmap)
    def __compute_triangle_areas(self, vertices, triangles):
        """Calculates the area of triangles making up a surface."""
        tri_u = vertices[triangles[:, 1], :] - vertices[triangles[:, 0], :]
        tri_v = vertices[triangles[:, 2], :] - vertices[triangles[:, 0], :]
        tri_norm = np.cross(tri_u, tri_v)
        triangle_areas = np.sqrt(np.sum(tri_norm ** 2, axis=1)) / 2.0
        triangle_areas = triangle_areas[:, np.newaxis]
        return triangle_areas
    def _compute_vertex_areas(self, vertices, triangles):
        triangle_areas = self.__compute_triangle_areas(vertices, triangles)
        vertex_areas = np.zeros((vertices.shape[0]))
        for triang, vertices in enumerate(triangles):
            for i in range(3):
                vertex_areas[vertices[i]] += 1./3. * triangle_areas[triang]
        return vertex_areas

    def __get_equilibrium(self, model, init):
        # the number of variables we need estimates for
        nvars = len(model.state_variables)
        cvars = len(model.cvar)
        def func(x):
            x = x[0:nvars]
            fx = model.dfun(x.reshape((nvars, 1, 1)),
                            np.zeros((cvars, 1, 1)))
            return fx.flatten()
        x = fsolve(func, init)
        return x

    def _computeinitcond(self, x0, num_regions):
        epileptor_equil = models.Epileptor()
        epileptor_equil.x0 = x0
        init_cond = self.__get_equilibrium(epileptor_equil, 
                        np.array([0.0, 0.0, 3.0, -1.0, 1.0, 0.0]))
        init_cond_reshaped = np.repeat(init_cond, num_regions).reshape((1, len(init_cond), num_regions, 1))
        self.init_cond = init_cond_reshaped
        return init_cond_reshaped

    def sample_randregions(self, num):
        randinds = np.random.randint(0, len(self.conn.region_labels), size=num).astype(int)
        regions = self.conn.region_labels[randinds]
        return randinds, regions

    def _getindexofregion(self, region):
        '''
        This is a helper function to determine the indices of the ez and pz region
        '''
        sorter = np.argsort(self.conn.region_labels)
        indice = sorter[np.searchsorted(self.conn.region_labels, region, sorter=sorter)].astype(int)
        return indice
