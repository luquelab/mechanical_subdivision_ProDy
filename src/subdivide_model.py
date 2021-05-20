from prody import *
import sys
import os
import numpy as np
import scipy
from scipy import sparse
from prody import LOGGER, SETTINGS
from sklearn import cluster

pdb = sys.argv[1]
filename = pdb + '.pdb'
n_clusters = int(sys.argv[2])

print(os.getcwd())
os.chdir("../results/models")
print(os.getcwd())
gnm = loadModel(pdb + '_full.gnm.npz')[:n_clusters]
calphas = loadAtoms('calphas_' + pdb + '.ag.npz')

test = cluster.SpectralClustering(assign_labels='kmeans',n_clusters=n_clusters,n_init=10,affinity='rbf',n_jobs=8).fit(gnm.getEigvecs())


domains = test.labels_
print(len(np.unique(domains)))
calphas.setData('domain',domains)
writePDB(pdb + '_domains.pdb',calphas,beta=domains)