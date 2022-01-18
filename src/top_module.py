#! ~/anaconda3/bin/python
#PBS -l nodes=2:ppn=24
#PBS -I walltime=02:00:00


from make_model import make_model
from subdivide_model import subdivide_model
from input import *


if rebuild_model:
    model, calphas = make_model(pdb, n_modes)

calphas, domains = subdivide_model(pdb, cluster_start, cluster_stop, cluster_step)
