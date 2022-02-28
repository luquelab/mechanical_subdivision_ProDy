import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, lobpcg
from memory_profiler import memory_usage, profile

@profile
def f():
    pdb = '3dkt'
    hess = sparse.load_npz('../results/models/' + pdb + 'hess.npz')
    a,b = eigsh(hess, k=10, sigma=1E-5, which='LA')
    return a

a = f()
# mem_usage = memory_usage(f)
# print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
# print('Maximum memory usage: %s' % max(mem_usage))


pdbs = ['1a34']

residues = []
sizes = []

for pdb in pdbs:
    hess = sparse.load_npz('../results/models/' + pdb + 'hess.npz')
    residues.append(int(hess.shape[0]/3))
    sizes.append((hess.data.nbytes + hess.indptr.nbytes + hess.indices.nbytes)/(10**6))
    mem_usage = memory_usage((sparse.eigsh, (hess,) {}))
#
# import matplotlib.pyplot as plt
#
# plt.scatter(residues,sizes)
# plt.show()