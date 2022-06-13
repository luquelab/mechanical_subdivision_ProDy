#! /home/ctbrown/anaconda3/bin/python3
#PBS -l nodes=1:ppn=24
#PBS -l walltime=96:00:00

import sys
sys.path.append('/home/ctbrown/mechanical_subdivisions/mechanical_subdivision_ProDy/src')

from make_model import make_model
from subdivide_model import subdivide_model
from settings import *
import time

print(pdb)

start = time.time()

if (mode =='full') or (mode =='hess') or (mode =='eigs'):
    make_model()
    subdivide_model()
elif (mode =='similarities') or (mode =='embedding') or (mode =='clustering'):
    subdivide_model()
else:
    print('Mode should be one of: full, hess, eigs, similarities, embedding, clustering')
    print('Defaulting to full')
    make_model()
    subdivide_model()

end = time.time()
print('Total time: ', end - start)
