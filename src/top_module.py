import sys
import scipy
from scipy import sparse
from prody import LOGGER, SETTINGS
from make_model import make_model
from subdivide_model import subdivide_model



pdb = sys.argv[1]
type = sys.argv[2]
n_modes = int(sys.argv[3])
cluster_range = sys.argv[4:]
cluster_range = [int(i) for i in cluster_range]


gnm, calphas = make_model(pdb, n_modes, type)

calphas, domains = subdivide_model(pdb, cluster_range)
