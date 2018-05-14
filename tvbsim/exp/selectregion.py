import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

class Regions(object):
	def __init__(self, regs, regs_xyz, epsilon):
		'''
		This will initialize the class object with the:
		- regs: list of region labels
		- regs_xyz: np.ndarray (nx3) of the region's xyz coords in T1 MRI space
		'''
		self.regs = regs
		self.regs_xyz = regs_xyz
		self.epsilon = epsilon

		# compute the pairwise distance matrix
		distmat = pdist(self.regs_xyz, metric='euclidean')
		self.distmat = distmat

	def generate_outsideset(self, set_of_regs):
		candidate_regs = list(set(list(self.regs)) - set(list(set_of_regs)))
		ind_of_regs = [i for i in range(len(self.regs)) if self.regs[i] in set_of_regs]
		outside_set = []

		# loop through all candidate regions
		for reg in candidate_regs:
			ireg = np.where(self.regs == reg)[0][0]

			# loop through all distances with the clinical set
			dists_of_reg = self.distmat[np.multiply(ireg,ind_of_regs)]
			for dist in dists_of_reg:
				if dist < self.epsilon:
					outside_set.append(ireg)
					break
		return outside_set

	def sample_outsideset(self, outside_set, numsamps):
		return np.random.choice(outside_set, size=numsamps, replace=False)

	