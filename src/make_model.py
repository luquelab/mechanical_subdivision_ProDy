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


def make_model(pdb, n_modes, mode):
    from input import cutoff, eigmethod, model

    capsid, calphas = getPDB(pdb)
    global anm, n_atoms, n_dim, n_asym
    n_atoms = calphas.getCoords().shape[0]
    n_dim = 3*n_atoms
    n_asym = int(n_atoms/60)

    if model=='gnm':
        anm = GNM(pdb + '_full')
    else:
        anm = ANM(pdb + '_full')

    if mode =='full':
        if model == 'gnm':
            kirch = buildKirch(pdb, calphas, cutoff)
        else:
            hess, kirch = buildHess(pdb, calphas, cutoff)
        evals, evecs = modeCalc(pdb, kirch, kirch, n_modes, eigmethod)
        evPlot(evals, evecs)
        distFlucts = distanceFlucts(calphas, evals, evecs, kirch, n_modes)
        sims = fluctToSims(distFlucts, pdb)
        saveAtoms(calphas, filename='../results/models/' + 'calphas_' + pdb)

    elif mode == 'hess':
        if not os.path.exists('../results/models/' + pdb + 'hess.npz') or not os.path.exists('../results/models/' + pdb + 'kirch.npz'):
            print('No hessian found. Building Hessian.')
            hess, kirch = buildHess(pdb, calphas, cutoff)
        else:
            hess, kirch = loadHess(pdb)
        evals, evecs = modeCalc(pdb, hess, kirch, n_modes, eigmethod)
        evPlot(evals, evecs)
        distFlucts = distanceFlucts(calphas, evals, evecs, kirch, n_modes)
        sims = fluctToSims(distFlucts, pdb)
        saveAtoms(calphas, filename='../results/models/' + 'calphas_' + pdb)
    elif mode == 'eigs':

        if not os.path.exists('../results/models/' + pdb + 'modes.npz'):
            print('No evecs found. Calculating')
            hess, kirch = buildHess(pdb, calphas, cutoff)
            evals, evecs = modeCalc(pdb, hess, kirch, n_modes, eigmethod)
        else:
            evals, evecs, kirch = loadModes(pdb, n_modes)

        evPlot(evals, evecs)
        distFlucts = distanceFlucts(calphas, evals, evecs, kirch, n_modes)
        sims = fluctToSims(distFlucts, pdb)
        saveAtoms(calphas, filename='../results/models/' + 'calphas_' + pdb)
    else:
        print('Not a valid mode, defaulting to full')
        hess, kirch = buildHess(pdb, calphas, cutoff)
        evals, evecs = modeCalc(pdb, hess, kirch, n_modes, eigmethod)
        evPlot(evals, evecs)
        distFlucts = distanceFlucts(calphas, evals, evecs, kirch, n_modes)
        sims = fluctToSims(distFlucts, pdb)
        saveAtoms(calphas, filename='../results/models/' + 'calphas_' + pdb)


    #from eigenCount import eigenCutoff

    #n_modes = eigenCutoff(evals, 0.001)
    #evals = evals[:n_modes]
    return sims, calphas


def getPDB(pdb):
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
    calphas = capsid.select('calpha').copy()
    print('Number Of Residues: ', calphas.getCoords().shape[0])
    os.chdir('../../src')

    writePDB('../results/subdivisions/' + pdb + '_ca.pdb', calphas,
             hybrid36=True)
    return capsid, calphas


def buildHess(pdb, calphas, cutoff=10.0):

    anm.buildHessian(calphas, cutoff=cutoff, kdtree=True, sparse=True)
    sparse.save_npz('../results/models/' + pdb + 'hess.npz', anm.getHessian())
    sparse.save_npz('../results/models/' + pdb + 'kirch.npz', anm.getKirchhoff())
    kirch = anm.getKirchhoff()

    return anm.getHessian(), kirch

def buildKirch(pdb, calphas, cutoff=10.0):

    anm.buildKirchhoff(calphas, cutoff=cutoff, kdtree=True, sparse=True)
    sparse.save_npz('../results/models/' + pdb + 'kirch.npz', anm.getKirchhoff())
    kirch = anm.getKirchhoff()

    return kirch


def loadHess(pdb):
    hess = sparse.load_npz('../results/models/' + pdb + 'hess.npz')
    anm._hessian = hess
    kirch = sparse.load_npz('../results/models/' + pdb + 'kirch.npz')
    return hess, kirch


def modeCalc(pdb, hess, kirch, n_modes, method):
    from input import model
    print('Calculating Normal Modes')
    start = time.time()

    if model=='anm':
        evals, evecs = eigsh(hess, k=n_modes, sigma=1e-8, which='LA')
    else:
        evals, evecs = eigsh(kirch, k=n_modes, sigma=1e-8, which='LA')
    print(evals)
    end = time.time()
    print(end - start)
    anm._eigvals = evals
    anm._n_modes = len(evals)
    anm._eigvecs = evecs
    anm._array = evecs
    saveModel(anm, filename='../results/models/' + pdb + 'gnm.npz')
    np.savez('../results/models/' + pdb + 'modes.npz', evals=evals, evecs=evecs)
    return evals, evecs


def loadModes(pdb, n_modes):
    from input import model
    anm = loadModel('../results/models/' + pdb + model + '.npz')
    print('Slicing Modes up to ' + str(n_modes))
    print()
    evals = anm.getEigvals()[:n_modes].copy()
    evecs = anm.getEigvecs()[:, :n_modes].copy()
    print(evecs.shape)
    kirch = sparse.load_npz('../results/models/' + pdb + 'kirch.npz')
    return evals, evecs, kirch


def evPlot(evals, evecs):
    import matplotlib.pyplot as plt
    print('Plotting')
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.scatter(np.arange(evals.shape[0]), evals, marker='D', label='eigs')
    fig.tight_layout()
    plt.show()

def sqfluctPlot(bfactors, sqFlucts):
    import matplotlib.pyplot as plt
    print('Plotting')
    bfactors = bfactors # * 3/(8 * np.pi**2)
    # sqFlucts = sqFlucts
    # sqFlucts = sqFlucts[:,np.newaxis]
    a, _, _, _ = np.linalg.lstsq(sqFlucts[:,np.newaxis], bfactors, rcond=-1)
    print(a)
    sqFlucts = a*sqFlucts
    stack = np.stack((bfactors,sqFlucts))
    print(stack.shape)
    print(np.corrcoef(bfactors,sqFlucts))
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.plot(np.arange(bfactors.shape[0])[:n_asym], np.abs(bfactors[:n_asym] - np.mean(bfactors))/np.std(bfactors), label='bfactors')
    ax.plot(np.arange(sqFlucts.shape[0])[:n_asym], np.abs(sqFlucts[:n_asym] - np.mean(sqFlucts))/np.std(sqFlucts), label='sqFlucts')
    ax.legend()
    fig.tight_layout()
    plt.show()


def distanceFlucts(calphas, evals, evecs, kirch, n_modes, fluctmode='direct'):
    print(evecs.shape[0] / n_atoms)
    from scipy import sparse
    from input import model
    if fluctmode == 'sample':
        from sampling import calcSample
        print('Sampling Method')
        coords = calphas.getCoords()
        d = calcSample(coords, evals, evecs, 50000, kirch.row, kirch.col)
        n_d = int(evecs.shape[0] / 3)
        d = sparse.coo_matrix((d, (kirch.row, kirch.col)), shape=(n_d, n_d))
    else:
        print('Direct Calculation Method')
        kirch = kirch.tocoo()
        covariance = sparse.lil_matrix((n_atoms, n_atoms))
        df = sparse.lil_matrix((n_atoms, n_atoms))
        if model=='anm':
            covariance = con_c(evals[:n_modes].copy(), evecs[:, :n_modes].copy(), covariance, kirch.row, kirch.col)
            covariance = covariance.tocsr()
        else:
            covariance = gCon_c(evals[:n_modes].copy(), evecs[:, :n_modes].copy(), covariance, kirch.row, kirch.col)
            covariance = covariance.tocsr()
            print(covariance.min())
        bfactors = calphas.getBetas()
        sqFlucts = covariance.diagonal()
        sqfluctPlot(bfactors, sqFlucts.copy())
        d = con_d(covariance, df, kirch.row, kirch.col)
        d = d.tocsr()
        d.eliminate_zeros()
        print(d.min())
    print('Average Fluctuations Between Elements', d.data.mean())
    fluctPlot(d)
    return d


def fluctPlot(d):
    import matplotlib.pyplot as plt
    print('Plotting Fluctuation Histogram')
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.hist(d.data, bins=50, histtype='stepfilled')
    fig.tight_layout()
    plt.show()


def fluctToSims(d, pdb):
    d = d.tocsr()
    d.eliminate_zeros()
    nnDistFlucts = np.mean(np.sqrt(d.data))

    sigma = 1 / (2 * nnDistFlucts ** 2)
    sims = -sigma * d ** 2
    data = sims.data
    data = np.exp(data)
    sims.data = data
    sparse.save_npz('../results/models/' + pdb + 'sims.npz', sims)

    return sims

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

@nb.njit(parallel=True)
def gCov(evals, evecs, i, j):
    n_e = evals.shape[0]
    n_d = evecs.shape[1]
    cov = 0
    for n in nb.prange(n_e):
        l = evals[n]
        cov += 1 / l * (evecs[i, n] * evecs[j, n])
    return cov

#@nb.njit()
def gCon_c(evals, evecs, c, row, col):
    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        c[i, j] = gCov(evals, evecs, i, j)
    return c



def con_c(evals, evecs, c, row, col):
    from pythranFuncs import cov
    n_d = int(evecs.shape[0] / 3)
    n_e = evals.shape[0]

    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        c[i, j] = cov(evals, evecs, i, j)
    return c

#@nb.njit()
def con_d(c, d, row, col):
    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        d[i, j] = np.abs(c[i,i] + c[j,j] - 2 * c[i, j])
    return d