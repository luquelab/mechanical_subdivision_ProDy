#! /home/ctbrown/anaconda3/bin/python3
#PBS -l nodes=1:ppn=24
#PBS -l walltime=96:00:00

if __name__ == '__main__':
    import sys
    sys.path.append('/home/ctbrown/mechanical_subdivisions/mechanical_subdivision_ProDy/src')

    from make_model import make_model
    from subdivide_model import subdivide_model
    from settings import *
    import time
    from memory_profiler import memory_usage

    print('Performing NMA and Quasi-Rigid Cluster Identification for PDB entry: ' + pdb)

    start = time.time()

    if (mode =='full') or (mode =='hess') or (mode =='eigs'):
        mem_usage = memory_usage( make_model)
        print('Peak memory usage: ', max(mem_usage))
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
