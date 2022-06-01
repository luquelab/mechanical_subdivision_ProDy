global mode
mode = 'full'

global pdb, pdbx
pdb = '2e0z'
pdbx = False

global model, fanm, cbeta, cutoff, d2, flexibilities
model = 'anm'
fanm = 0.1
cbeta = False
d2 = False
flexibilities = False
cutoff = 10

global n_modes, fitmodes, eigmethod
n_modes = 300
fitmodes = True
eigmethod = 'lobcuda'



global cluster_method, scoreMethod, cluster_start, cluster_stop, cluster_step
cluster_method = 'discretize'
scoreMethod = 'mean'
cluster_start = 4
cluster_stop = 40
cluster_step = 2