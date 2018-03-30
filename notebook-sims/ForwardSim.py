from tvb.simulator.lab import *
import os.path
from matplotlib import colors, cm
import matplotlib.pyplot as pyplt
import time
import scipy.signal as sig
import scipy.spatial.distance as dists
import numpy as np
from InitialConditions import get_equilibrium
import time


def forwardSim(ez,ezType,pz,noiseON,simLen,savePath):
  project_dir = "/home/anirudh/Academia/Projects/VEP/data/CJ"
  con = connectivity.Connectivity.from_file(os.path.join(project_dir, "connectivity.zip"))
  con.speed = np.inf
  # normalize
  con.weights = con.weights/np.max(con.weights)
  num_regions = len(con.region_labels)

#  pyplt.figure()
#  image = con.weights
#  norm = colors.LogNorm(1e-7, image.max()) #, clip='True')
#  pyplt.imshow(image, norm=norm, cmap=cm.jet)
#  pyplt.colorbar()
#  #max(con.weights[con.weights != 0])


  epileptors = models.Epileptor(variables_of_interest=['x1', 'y1', 'z', 'x2', 'y2', 'g', 'x2 - x1'])
  epileptors.r = 0.0001
  epileptors.Ks = np.ones(num_regions)*(-1.0)*20.0


  # Patient specific modifications
  #ez = [9]
  #pz = [6, 27]

  epileptors.x0 = np.ones(num_regions)*-2.3
  if(noiseON):
    epileptors.x0[ez] = -1.7
  else:
    epileptors.x0[ez] = -1.3

  epileptors.x0[pz] = -2.05


  coupl = coupling.Difference(a=1.)

  nsf = 1 # noise scaling factor
  hiss = noise.Additive(nsig = nsf*np.array([0.01, 0.01, 0., 0.00015, 0.00015, 0.]))
  if(noiseON):
    heunint = integrators.HeunStochastic(dt=0.04, noise=hiss)
  else:
    heunint = integrators.HeunDeterministic(dt=0.04)



  #mon_raw = monitors.Raw()
  mon_tavg = monitors.TemporalAverage(period=1.0)
  mon_SEEG = monitors.iEEG.from_file(sensors_fname=os.path.join(project_dir, "seeg.txt"),
                                     projection_fname=os.path.join(project_dir, "gain_inv-square.txt"),
                                     period=1.0,
                                     variables_of_interest=[6]
                                     )
  num_contacts = mon_SEEG.sensors.labels.size


  con.cortical[:] = True     # To avoid adding analytical gain matrix for subcortical sources

# Find a fixed point to initialize the epileptor in a stable state
  epileptor_equil = models.Epileptor()
  epileptor_equil.x0 = -2.3
#init_cond = np.array([0, -5, 3, 0, 0, 0])
  init_cond = get_equilibrium(epileptor_equil, np.array([0.0, 0.0, 3.0, -1.0, 1.0, 0.0]))
  init_cond_reshaped = np.repeat(init_cond, num_regions).reshape((1, len(init_cond), num_regions, 1))
  sim = simulator.Simulator(model=epileptors,
                            initial_conditions=init_cond_reshaped,
                            connectivity=con,
                            coupling=coupl,
                            conduction_speed=np.inf,                          
                            integrator=heunint,
                            monitors=[mon_tavg, mon_SEEG])

  sim.configure()


  (ttavg, tavg), (tseeg, seeg) = sim.run(simulation_length=simLen)


#  # Normalize the time series to have nice plots
#  tavgn = (tavg - np.min(tavg,0))/(np.max(tavg, 0) - np.min(tavg, 0))
  seegn = (seeg - np.min(seeg,0)) / (np.max(seeg,0) - np.min(seeg,0))
#  seegn = seegn - np.mean(seegn)
#  b, a = sig.butter(2, 0.1, btype='highpass', output='ba')
#  seegf = sig.filtfilt(B, A, seegn)
#  seegf = np.zeros(seegn.shape)
#  for i in range(num_contacts):
#      seegf[:, 0, i, 0] = sig.filtfilt(b, a, seeg[:, 0, i, 0])

  # Save data for data fitting
  savePathTS=os.path.join(savePath,"timeSeries_ezType=%s_nez=%d_npz=%d_noise=%s"%(ezType,np.size(ez),np.size(pz),str(noiseON)))
  np.savez(savePathTS,ttavg=ttavg,tavg=tavg,tseeg=tseeg,seeg=seeg,
           ezType=ezType,ez=ez,pz=pz,noiseON=noiseON)

#  #Plot raw time series
  pyplt.figure(figsize=(9,10))

  indf = 0
  indt = -1

  regf = 0
  regt = 84

#  pyplt.plot(ttavg, tavgn[:, 6, regf:regt, 0] + np.r_[regf:regt], 'r')
#  pyplt.yticks(np.r_[regf:3.5*regt:3.5], con.region_labels[regf:regt])
  pyplt.plot(ttavg[:], tavg[:, 6, regf:regt, 0]/4 + np.r_[regf:regt], 'r')
  pyplt.yticks(np.r_[regf:regt], np.r_[regf:regt])
  pyplt.title("Epileptors time series")
  pyplt.tight_layout()
  pyplt.savefig(os.path.join(savePath,"sources_ezType=%s_nez=%d_npz=%d_noise=%s.png"%(ezType,np.size(ez),np.size(pz),str(noiseON))))
#  pyplt.show(block=False)
  idxs_bplr_excld = np.array([6,20,28,42,56,70,83,96,109,122,136,148])
  seegn_bplr = seegn[:, 0, 1:num_contacts, 0] - seegn[:, 0, 0:num_contacts-1, 0] 
  seegn_bplr = seegn_bplr[:,list(set(np.r_[0:num_contacts-1]) - set(idxs_bplr_excld))]
  
  pyplt.figure(figsize=(9,10))
  pyplt.plot(tseeg[:], seegn_bplr + np.r_[:np.shape(seegn_bplr)[1]])
#  pyplt.plot(tseeg[:], (seeg[:, 0, 1:num_contacts, 0] - seeg[:, 0, 0:num_contacts-1, 0])/np.max(seeg) + np.r_[0:2*(num_contacts-1):2])

#  pyplt.plot(tseeg[:], (seegn[:, 0, 1:num_contacts, 0] - seegn[:, 0, 0:num_contacts-1, 0]) + np.r_[0:num_contacts-1])
  sensor_labels_bplr = np.array([mon_SEEG.sensors.labels[i+1]+'-'+\
                       mon_SEEG.sensors.labels[i] for i in xrange(mon_SEEG.sensors.labels.size - 1)])
  sensor_labels_bplr = sensor_labels_bplr[list(set(np.r_[0:num_contacts-1]) - set(idxs_bplr_excld))]

  pyplt.yticks(np.r_[:np.size(sensor_labels_bplr)], sensor_labels_bplr)
#  pyplt.yticks(np.r_[0:2*(num_contacts-1):2], mon_SEEG.sensors.labels[:])

  pyplt.title("SEEG")
  pyplt.tight_layout()
  pyplt.savefig(os.path.join(savePath,"seeg_ezType=%s_nez=%d_npz=%d_noise=%s.png"%(ezType,np.size(ez),np.size(pz),str(noiseON))))
#  pyplt.show(block=False)

#  pyplt.figure(figsize=(10, 6))
#
#  electrodes = [("FCA'", 7), ("GL'", 7), ("CU'", 6), ("PP'", 1),
#                ("PI'", 5), ("GC'", 8), ("PFG'", 10),
#                ("OT'", 5), ("GPH'", 6), ("PFG", 10)]
#
#
#  for i, (el, num) in enumerate(electrodes):
#      ind = np.where(mon_SEEG.sensors.labels == el + str(num))[0][0]
#      pyplt.plot(tseeg[:], (seegn[:, 0, ind, 0] - seegn[:, 0, ind - 1, 0])/0.5 + i)
#
#  labels = [el[0] + str(el[1]) + "-" + str(el[1] - 1) for el in electrodes]
#  pyplt.yticks(np.r_[:len(electrodes)], labels)
#  pyplt.tight_layout()
#  pyplt.show()
#  pyplt.close("all")

#def choosePZ(ez,npz):
#  pz = []
#  for i in xrange(npz):
#    while(True):
#      tpz = np.random.randint(0,84)
#      if(tpz in ez):
#        continue
#      else:
#        break
#    pz.append(tpz)
#  return pz
#

## Perform forward Simulations with randomly chosez EZ and PZ
#for nez in xrange(1,4):
#  for npz in xrange(0,5):
#    if(nez == 1):
#      ez = [np.random.randint(0,84)]
#    else:
#      ez = np.random.randint(0,84,nez)
#    pz = choosePZ(ez,npz)
#    simLen = 10*1000
#    savePath = "../../results/ForwardSim/CJ/run2"
#    tstart = time.time()
#    forwardSim(ez=ez,pz=pz,noiseON=True,simLen=simLen,savePath=savePath)
#    forwardSim(ez=ez,pz=pz,noiseON=False,simLen=simLen,savePath=savePath)
#    telapsed = (time.time()-tstart)/60.0
#    print("Elapsed time for %d-ez and %d-pz: %.2f mins"%(nez,npz,telapsed))

## Test forward simulations
#forwardSim(ez=[63],ezType='small',pz=[],noiseON=False,simLen=10000,savePath='/home/anirudh/Documents/')
#forwardSim(ez=[63],ezType='small',pz=[],noiseON=True,simLen=10000,savePath='/home/anirudh/Documents/')

 Perform forward simulations with choice of EZ based on effect
 on sensors and choice of PZ based on the metric argmax(j){Cji/Dji}

gain_mat = np.loadtxt('../../data/CJ/gain_inv-square.txt')
gm_sum = np.sum(gain_mat,0)
struct_con = connectivity.Connectivity.from_file(os.path.join\
              ("/home/anirudh/Academia/Projects/VEP/data/CJ","connectivity.zip"))
weights = struct_con.weights
roi_dists = dists.cdist(struct_con.centres,struct_con.centres,'euclidean')


EZ = dict()
#EZ['large'] = np.flip(np.argsort(gm_sum)[-5:],0)
#EZ['medium'] = np.flip(np.argsort(gm_sum)[45:50],0)
#EZ['small'] = np.flip(np.argsort(gm_sum)[0:5],0)
EZ['large'] = np.array(np.argsort(gm_sum)[-1])
EZ['medium'] = np.array(np.argsort(gm_sum)[45])
EZ['small'] = np.array(np.argsort(gm_sum)[0])
print("EZ_large:%s\nEZ_medium:%s\nEZ_small:%s"%(EZ['large'],EZ['medium'],EZ['small']))

# Select PZ
pz_metric = weights/(2*roi_dists)
simLen = 10000 # simulation length in milliseconds
savePath = "../../results/ForwardSim/CJ/run3"
for ez_type in ['large','medium','small']:
  for nez in xrange(1,4):
    ez_prim = EZ[ez_type]
    ez_metric = dists.cdist(struct_con.centres[ez_prim,:].reshape(1,3),struct_con.centres)
    ez = np.argsort(ez_metric)[0,0:nez]
        
    for npz in xrange(5):
      if(npz == 0):
        pz = np.array([],dtype=int)
      else:
        pzm = np.sum(pz_metric[:,ez],1)
        pz = np.argsort(pzm)[-npz-nez:-nez]
      print("ez=%s\npz=%s"%(ez,pz))
      print("gmsum(ez):%s\ngmsum(pz):%s\n"%(gm_sum[ez],gm_sum[pz]))
      forwardSim(ez=ez,ezType=ez_type,pz=pz,noiseON=True,simLen=simLen,savePath=savePath)
      forwardSim(ez=ez,ezType=ez_type,pz=pz,noiseON=False,simLen=simLen,savePath=savePath)

