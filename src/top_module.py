import sys
import scipy
from scipy import sparse
from prody import LOGGER, SETTINGS
from make_model import make_model
from subdivide_model import subdivide_model

pdb = sys.argv[1]
n_modes = int(sys.argv[2])
n_clusters = int(sys.argv[3])

gnm, calphas = make_model(pdb, n_modes)

calphas, domains = subdivide_model(pdb, n_clusters)
