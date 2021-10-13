from prody import *
import os
import wget
import shutil
import gzip
from scipy.sparse.linalg import eigsh
import time
import psutil


def make_model(pdb, n_modes, type):
    filename = pdb + '_full.pdb'


    os.chdir("../data/capsid_pdbs/")

    if not os.path.exists(filename):
        vdb_url = 'https://files.rcsb.org/download/' + pdb + '.pdb.gz'
        print(vdb_url)
        vdb_filename = wget.download(vdb_url)
        with gzip.open(vdb_filename, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    capsid = parsePDB(filename, biomol=True)
    calphas = capsid.select('ca').copy()

    if type == 'gnm':

        gnm = GNM(pdb + '_full')
        gnm.buildKirchhoff(calphas, cutoff=7.5, kdtree=False, sparse=True)
        print('Memory Usage: ', psutil.virtual_memory().percent)
        gnm.calcModes(n_modes,turbo=True)

        print(os.getcwd())
        os.chdir("../../results/models")
        print(os.getcwd())
        saveModel(gnm,matrices=True)
        saveAtoms(calphas,filename='calphas_' + pdb)

        return gnm, calphas

    elif type == 'anm':
        anm = ANM(pdb + '_full')
        anm.buildHessian(calphas, cutoff=10.0, kdtree=True, sparse=True)
        print('Calculating Normal Modes')
        start = time.time()
        evals, evecs = eigsh(anm.getHessian(), k=n_modes, sigma=1E-5, which='LA')
        end = time.time()
        print(end - start)
        anm._eigvals = evals
        anm._n_modes = len(evals)
        anm._eigvecs = evecs
        anm._vars = 1 / evals
        anm._array = evecs

        print(os.getcwd())
        os.chdir("../../results/models")
        print(os.getcwd())
        print('Saving Model Results')
        saveModel(anm, matrices=True)
        saveAtoms(calphas, filename='calphas_' + pdb)

        return anm, calphas
    else:
        raise ValueError('type must be anm or gnm.')
