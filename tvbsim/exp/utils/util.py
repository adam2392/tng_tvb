import numpy as np 
import random

def randshuffleweights(weights):
    weights = np.random.choice(weights.ravel(), size=weights.shape, replace=False)
    return weights

def randshufflepats(patientlist, patient):
    patientlist.remove(patient)
    randpat = random.choice(patientlist)
    return randpat