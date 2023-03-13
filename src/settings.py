# mode: Different options for where the program begins
#     full - start from a PDB and do the entire process
#     hess - start from an already computed hessian
#     eigs - start from an already computed set of eigenvectors/values
#     similarities - start from an already computed set of similarities
#     embedding - start from an already computed spectral embedding
#     clustering - start from an already computed clustering (rigidity analysis and plotting)

mode = 'full'

# PDB variables
# pdb: The protein databank id for downloading the file
# pdbx: true or false, whether the pdb uses the pdbx/mmcif format
# chains: chains to include in NMA/ENM

global pdb, pdbx, local, chains
pdb = '6c21'
pdbx = False
local = False
chains = ''

# Model Parameters
# model: Anisotropic or Gaussian Network Model
#     'anm' - Anisotropic Network Model
#     'gnm' - Gaussian Network Model
# cutoff: Distance (in angstroms) beyond which residues will not be connected
# fanm: Degree of anisotropy when model='anm'. 0.0 is equivalent to gaussian network model, 1.0 is standard ANM
# cbeta: Whether to include beta carbon atoms
# d2: Whether to scale interaction strength by the inverse square distance
# flexibilities: Whether to use b-factors from the pdb to estimate pairwise spring constants

global model, fanm, cbeta, aaGamma, bbGamma, abGamma, cutoff, d2, flexibilities, backboneStrength, bblen
model = 'gnm'
cutoff = 7.5

fanm = 0
baseDistance = 1
gfunc = 'power'
d2 = 0
flexibilities = False

backboneConnect = False
backboneStrength = 3
bblen = 3

cbeta = True
aaGamma = 2
bbGamma = 1
abGamma = 1

# NMA Options
# n_modes: Number of low frequency modes to calculate
# fitmodes: whether to select the number of modes (less than n_modes) that best fit b-factors and use those modes in
# future calculations
# eigmethod: The eigensolver to use when performing NMA
#     'eigsh' - Scipy sparse eigensolver using ARPACK's implementation of the Implicitly Restarted Lanczos Method
#     'lobpcg' - Scipy sparse eigensolver using the Locally Optimal Block Preconditioned Conjugate Gradient Method. Less accurate and less memory cost.
#     'lobcuda' - GPU accelerated lobpcg through the cupy package, requires a cuda implementation

global n_modes, fitmodes, eigmethod
n_modes = 300
fitmodes = True
intercept = False
eigmethod = 'lobcuda'

# Distance Fluctuation Options
d_cutoff = 7.5

# Clustering Options
# cluster_method: Method of determining quasi-rigid clusters from spectral embedding
#     'kmeans' - K-means clustering using the sklearn package
#     'discretize' - sklearn implementation of algorithm from "Multiclass Spectral Clustering". Faster, more symmetric,
#                    and more accurate than kmeans ...
# scoreMethod: How to calculate the cluster quality scores
#     'mean' - use the mean value of the scores of each residue
#     'median' - - use the median value of the scores of each residue
global cluster_method, scoreMethod, cluster_start, cluster_stop, cluster_step, calc_rigidities, chains_clust
cluster_method = 'discretize'
scoreMethod = 'median'
cluster_start = 12
cluster_stop = 244
cluster_step = 2
calc_rigidities = False
chains_clust = ''