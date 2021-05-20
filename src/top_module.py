from prody import *
import sys
import os
import numpy as np
import scipy
from scipy import sparse
from prody import LOGGER, SETTINGS
from sklearn import cluster
from make_model import make_model
pdb = sys.argv[0]
gnm, calpha = make_model(pdb, n_modes)

gnm, calpha = make_model(pdb, n_modes)