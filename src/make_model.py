from prody import *
import os
import wget
import shutil
import gzip
from scipy.sparse.linalg import eigsh
import time
import numba as nb
import numpy as np
from scipy import sparse

def make_model(pdb, n_modes):
    os.chdir('../data/capsid_pdbs')
    filename = pdb + '_full.pdb'
    if not os.path.exists(filename):
        vdb_url = 'https://files.rcsb.org/download/' + pdb + '.pdb.gz'
        print(vdb_url)
        vdb_filename = wget.download(vdb_url)
        with gzip.open(vdb_filename, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    capsid = parsePDB(filename, biomol=True)
    calphas = capsid.select('ca').copy()

    os.chdir('../../src')

    anm = ANM(pdb + '_full')
    anm.buildHessian(calphas, cutoff=10.0, kdtree=True, sparse=True)
    sparse.save_npz('../results/models/' + pdb + 'hess.npz', anm.getHessian())
    print('Calculating Normal Modes')
    start = time.time()
    evals, evecs = eigsh(anm.getHessian(), k=n_modes, sigma=1E-5, which='LA')
    print(evals)
    end = time.time()
    print(end - start)
    anm._eigvals = evals
    anm._n_modes = len(evals)
    anm._eigvecs = evecs
    anm._array = evecs

    import matplotlib.pyplot as plt
    print('Plotting')
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.scatter(np.arange(evals.shape[0]), evals, marker='D', label='eigs')
    fig.tight_layout()
    plt.show()

    model = anm


    n_d = int(evecs.shape[0] / 3)

    kirch = model.getKirchhoff().tocoo()

    covariance = sparse.lil_matrix((n_d, n_d))
    df = sparse.lil_matrix((n_d, n_d))
    covariance = con_c(evals, evecs, covariance, kirch.row, kirch.col)
    covariance = covariance.tocsr()
    d = con_d(covariance, df, kirch.row, kirch.col)
    d = d.tocsr()

    nnDistFlucts = np.mean(d.data)

    sigma = 1 / (2 * nnDistFlucts ** 2)
    sims = -sigma * d ** 2
    data = sims.data
    data = np.exp(data)
    sims.data = data

    sparse.save_npz('../results/models/' + pdb + 'sims.npz', sims)

    saveAtoms(calphas, filename='../results/models/' + 'calphas_' + pdb)

    return sims, calphas

@nb.njit(parallel=True)
def cov(evals, evecs, i, j):
    n_e = evals.shape[0]
    n_d = evecs.shape[1]
    tr1 = 0
    tr2 = 0
    tr3 = 0
    for n in nb.prange(n_e):
        l = evals[n]
        tr1 += 1 / l * (evecs[3 * i, n] * evecs[3 * j, n] + evecs[3 * i + 1, n] * evecs[3 * j + 1, n] + evecs[
            3 * i + 2, n] * evecs[3 * j + 2, n])
        # tr2 += 1 / l * (evecs[3 * i, n] * evecs[3 * i, n] + evecs[3 * i + 1, n] * evecs[3 * i + 1, n] + evecs[
        #     3 * i + 2, n] * evecs[3 * i + 2, n])
        # tr3 += 1 / l * (evecs[3 * j, n] * evecs[3 * j, n] + evecs[3 * j + 1, n] * evecs[3 * j + 1, n] + evecs[
        #     3 * j + 2, n] * evecs[3 * j + 2, n])
    cov = tr1  # / np.sqrt(tr2 * tr3)
    return cov

def con_c(evals, evecs, c, row, col):
    n_d = int(evecs.shape[0] / 3)
    n_e = evals.shape[0]

    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        c[i, j] = cov(evals, evecs, i, j)
    return c

def con_d(c, d, row, col):
    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        d[i, j] = 2 - 2 * c[i, j]
    return d