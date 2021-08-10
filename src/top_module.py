import sys
import scipy
from scipy import sparse
from prody import LOGGER, SETTINGS
from make_model import make_model
from subdivide_model import subdivide_model

pdb = sys.argv[1]
n_modes = int(sys.argv[2])
n_cluster_min = int(sys.argv[3])
n_cluster_max = int(sys.argv[4])
type = sys.argv[5]

gnm, calphas = make_model(pdb, n_modes, type)

calphas, domains = subdivide_model(pdb, n_cluster_min, n_cluster_max)
