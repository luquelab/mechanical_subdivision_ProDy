
# Program modes
# full: start from a PDB and do the entire process
# hess: start from an already computed hessian
# eigs: start from an already computed set of eigenvectors/values
# similariies: start from an already computed set of similarities
# embedding: start from an already computed spectral embedding
# clustering: start from an already computed clustering (rigidity analysis and plotting)
global mode
mode = 'eigs'

global pdb, pdbx
pdb = '2e0z'
pdbx = False

global model, fanm, cbeta, cutoff, d2, flexibilities
model = 'gnm'
fanm = 0.1
cbeta = False
d2 = False
flexibilities = False
cutoff = 7.5

global n_modes, fitmodes, eigmethod
n_modes = 900
fitmodes = True
eigmethod = 'lobpcg'



global cluster_method, scoreMethod, cluster_start, cluster_stop, cluster_step
cluster_method = 'discretize'
scoreMethod = 'mean'
cluster_start = 12
cluster_stop = 70
cluster_step = 2