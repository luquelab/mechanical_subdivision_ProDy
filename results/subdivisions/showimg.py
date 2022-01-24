# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# fig, ax = plt.subplots(figsize=(18,10))
# plt.imshow(mpimg.imread('5muu_420_domains.png'))
# plt.show()

from prody import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
pdb = '6xgq'
nc = 212
cm = plt.get_cmap("viridis")

results = np.load('./' + pdb + '/' + pdb + '_' + str(nc) + '_results.npz')
labels = results['labels']
calphas = loadAtoms('../models/' + 'calphas_' + pdb + '.ag.npz')
coords = calphas.getCoords()

fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
ax3D.scatter(coords[:,0], coords[:,1], coords[:,2], s=10, c=labels, marker='o')
plt.show()