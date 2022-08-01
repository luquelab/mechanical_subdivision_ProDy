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
from settings import *


def make_model():

    if pdbx:
        capsid, calphas, coords, bfactors, title = getPDBx(pdb)
    else:
        capsid, calphas, title = getPDB(pdb)
        coords = calphas.getCoords()
        bfactors = calphas.getBetas()

    print(title)
    print('# Of Residues: ', coords.shape)

    if mode == 'full':
        hess, kirch = buildModel(pdb, calphas, coords, bfactors, cutoff)
    else:
        if model == 'gnm':
            kirch = loadKirch(pdb)
            hess = kirch
        else:
            hess, kirch = loadHess(pdb)

    if mode == 'eigs':
        evals, evecs, kirch = loadModes(pdb, n_modes)
    else:
        evals, evecs = modeCalc(pdb, hess, kirch, n_modes, eigmethod, model)

    evPlot(evals, evecs, title)
    nm, gamma = mechanicalProperties(bfactors, evals, evecs, title)

    distFlucts = distanceFlucts(evals, evecs, kirch, n_modes, title)
    sims = fluctToSims(distFlucts, pdb)

    # return -1


def getPDB(pdb):
    from settings import pdbx
    os.chdir('../data/capsid_pdbs')
    if pdbx:
        filename = pdb + '_full.cif'
    else:
        filename = pdb + '_full.pdb'
    if not os.path.exists(filename):
        pdb_url = 'https://files.rcsb.org/download/' + pdb
        if pdbx:
            pdb_url = pdb_url + '.cif.gz'
        else:
            pdb_url = pdb_url + '.pdb.gz'
        print(pdb_url)
        pdb_filename = wget.download(pdb_url)
        with gzip.open(pdb_filename, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    if pdbx:
        capsid, header = parseMMCIF(filename, biomol=True, header=True)
        print(capsid.getTitle())
    else:
        capsid, header = parsePDB(filename, header=True, biomol=True, secondary=True, extend_biomol=True)
        print(capsid)
    if type(capsid) is list:
        capsid = capsid[0]
    capsid = capsid.select('protein').copy()
    capsid = addNodeID(capsid)
    calphas = capsid.select('protein and name CA')
    print('Number Of Residues: ', calphas.getCoords().shape[0])
    os.chdir('../../src')

    writePDB('../results/subdivisions/' + pdb + '_ca_prot.pdb', calphas,
             hybrid36=True)
    title = header['title']

    return capsid, calphas, title

def getPDBx(pdb):
    import biotite.database.rcsb as rcsb
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io as strucio
    file_name = rcsb.fetch(pdb, "pdbx", target_path="../data/capsid_pdbs")
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(file_name)
    capsid = pdbx.get_assembly(pdbx_file, assembly_id="1", model=1, extra_fields=['b_factor'])
    title = pdbx_file.get_category('pdbx_database_related')['details']
    print(title)
    print("Number of protein chains:", struc.get_chain_count(capsid))
    calphas = capsid[capsid.atom_name == 'CA']
    coords = calphas.coord
    return capsid, calphas, coords, calphas.b_factor, title


def addNodeID(atoms):
    atoms.setData('nodeid', 1)
    atoms.setData('chainNum', 0)
    i = 0
    chid = 0
    ch0 = atoms[0].getChid()
    seg0 = atoms[0].getSegname()
    for at in atoms.iterAtoms():
        if at.getName() == 'CA':
            if at.getChid() == ch0 and at.getSegname() == seg0:
                at.setData('chainNum', chid)
            else:
                chid += 1
            ch0 = at.getChid()
            seg0 = at.getSegname()

            at.setData('nodeid', i)

            i += 1
        else:
            at.setData('nodeid', i)

    return atoms

def buildModel(pdb, calphas, coords, bfactors, cutoff=10.0):
    from ENM import buildENM, betaCarbonModel
    from settings import cbeta
    start = time.time()
    if cbeta:
        kirch, hess = betaCarbonModel(calphas)
    else:
        kirch, hess = buildENM(calphas, coords, bfactors)
    # anm.setHessian(hess)
    # anm._kirchhoff = kirch
    sparse.save_npz('../results/models/' + pdb + 'hess.npz', hess)
    sparse.save_npz('../results/models/' + pdb + 'kirch.npz', kirch)
    end = time.time()
    print('Hessian time: ', end - start)
    return hess, kirch


def loadHess(pdb):
    hess = sparse.load_npz('../results/models/' + pdb + 'hess.npz')
    # anm._hessian = hess
    kirch = sparse.load_npz('../results/models/' + pdb + 'kirch.npz')
    return hess, kirch


def loadKirch(pdb):
    kirch = sparse.load_npz('../results/models/' + pdb + 'kirch.npz')
    return kirch


def modeCalc(pdb, hess, kirch, n_modes, eigmethod, model):
    # from input import model#, eigmethod
    print('Calculating Normal Modes')
    start = time.time()

    cuth_mkee = False
    if model == 'anm':
        mat = hess
    else:
        mat = kirch

    if cuth_mkee:
        from scipy.sparse.csgraph import reverse_cuthill_mckee as rcm
        perm = rcm(mat, symmetric_mode=True)
        mat.indices = perm.take(mat.indices)
        mat = mat.tocsc()
        mat.indices = perm.take(mat.indices)
        mat = mat.tocsr()
        print(perm)
    useMass = False
    n_dim = mat.shape[0]
    if eigmethod == 'eigsh':
        M = sparse.identity(n_dim)
        evals, evecs = eigsh(mat, M=M, k=n_modes, sigma=1e-8, which='LA')
    elif eigmethod == 'lobpcg':
        from scipy.sparse.linalg import lobpcg
        print(mat.shape)
        epredict = np.random.rand(n_dim, n_modes + 6)
        evals, evecs = lobpcg(mat, epredict, largest=False, tol=0, maxiter=n_dim)
        evals = evals[6:]
        evecs = evecs[:, 6:]
        print(evecs.shape)
    elif eigmethod == 'lobcuda':
        import cupy as cp
        from cupyx.scipy.sparse.linalg import lobpcg as clobpcg
        sparse_gpu = cp.sparse.csr_matrix(mat.astype(cp.float32))
        epredict = cp.random.rand(n_dim, n_modes + 6, dtype = np.float32)
        print(sparse_gpu, epredict)
        M = cp.sparse.identity(n_dim)
        evals, evecs = clobpcg(sparse_gpu, epredict, largest=False, tol=0)
        evals = cp.asnumpy(evals[6:])
        evecs = cp.asnumpy(evecs[:, 6:])
        print(evecs.shape)
    if cuth_mkee:
        evecs = evecs[perm, :].copy()
    useMass=False
    end = time.time()
    print('NMA time: ', end - start)
    np.savez('../results/models/' + pdb + model + 'modes.npz', evals=evals, evecs=evecs)
    return evals, evecs


def loadModes(pdb, n_modes):
    from settings import model
    if model == 'anm':
        filename = '../results/models/' + pdb + model + 'modes.npz'
        if not os.path.exists(filename):
            filename = '../results/models/' + pdb + 'modes.npz'
        print(filename)
        modes = np.load(filename)
        evals = modes['evals'][:n_modes].copy()
        evecs = modes['evecs'][:, :n_modes].copy()
    else:
        modes = np.load('../results/models/' + pdb + model + 'modes.npz')
        evals = modes['evals'][:n_modes].copy()
        evecs = modes['evecs'][:, :n_modes].copy()
        # model = loadModel('../results/models/' + pdb + model + 'modes.npz')
        # evals = model.getEigvals()[:n_modes].copy()
        # evecs = model.getEigvecs()[:, :n_modes].copy()
    print('Loading ' + model + ' Modes')
    print('Slicing Modes up to ' + str(n_modes))

    print(evecs.shape)
    kirch = sparse.load_npz('../results/models/' + pdb + 'kirch.npz')
    return evals, evecs, kirch


def evPlot(evals, evecs, title):
    # from prody import writeNMD
    import matplotlib.pyplot as plt
    import matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 11}
    matplotlib.rc('font', **font)
    print('Plotting')
    ax.set_ylabel(r'$\omega^{2}$', fontsize=12)
    ax.set_xlabel('Smallest Eigenvalues', fontsize=12)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.legend()
    fig.suptitle(
        'Eigenvalues/Squared Frequencies: ' + title.title() + ' (' + pdb + ')', fontsize=12)

    ax.scatter(np.arange(evals.shape[0]), evals, marker='D', s=10, label='eigs')
    fig.tight_layout()
    plt.savefig('../results/subdivisions/' + pdb + '_evals.png')
    plt.show()


def icoEvPlot(evals, evecs, sqFlucts, title, coords):
    import matplotlib.pyplot as plt
    from settings import pdb
    uniques, inds, counts = np.unique(evals.round(decimals=8), return_index=True, return_counts=True)
    icoEvalInds = inds[counts == 1]
    print(icoEvalInds)
    icoEvals = evals[icoEvalInds]
    icoEvecs = evecs[:, icoEvalInds]
    anm = ANM(pdb)
    anm._eigvals = evals
    anm._eigvecs = evecs
    anm._array = evecs
    anm._n_modes = evals.shape[0]
    anm._vars = 1 / evals
    print(icoEvecs.shape)
    print(icoEvals)
    anm._n_atoms = evecs.shape[0]
    # bbanm, bbatoms = extendModel(anm, calphas, capsid.select('backbone'))
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.scatter(np.arange(icoEvals.shape[0]), icoEvals, marker='D', label='eigs')
    plt.show()
    #from nmdOut import nmdWrite
    # nmdWrite(pdb, coords, icoEvecs, sqFlucts)
    #writeNMD(pdb + '_ico.nmd', anm[icoEvalInds], calphas)
    from prody import traverseMode, writePDB
    # ensemble = traverseMode(anm[icoEvalInds[0]], calphas, n_steps=15, rmsd=12)
    # writePDB(pdb + '_icomode.pdb', ensemble, beta=sqFlucts)


def mechanicalProperties(bfactors, evals, evecs, title):
    from settings import pdb
    import matplotlib
    import matplotlib.pyplot as plt
    from bfactorFit import fluctFit
    # from score import collectivity, meanCollect, effectiveSpringConstant, overlapStiffness, globalPressure
    # _, calphas, title = getPDB(pdb)

    # from settings import cbeta
    # if cbeta:
    #     names = calphas.getNames()
    #
    #     ab = np.array([True if x == 'CA' else False for x in names])
    #     print(evecs.shape)
    #     ev = np.reshape(evecs, (-1,3,evals.shape[0]))
    #     evecs = ev[ab].reshape(evecs.shape)
    #     bfactors = bfactors[ab]
    print(evecs.shape)

    print('Plotting')
    nModes, coeff, k, sqFlucts, stderr = fluctFit(evals, evecs, bfactors)
    # icoEvPlot(evals, evecs, sqFlucts, title)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 11}
    matplotlib.rc('font', **font)

    gamma = (8 * np.pi ** 2) / k
    stderr =  stderr/k

    if model == 'anm':
        gamma = gamma / 3

    stderr = gamma * stderr
    from bfactorFit import confidenceInterval
    ci = confidenceInterval(bfactors, stderr)

    print(nModes, coeff, gamma)
    n_asym = int(bfactors.shape[0]/60)
    np.savez('../results/subdivisions/' + pdb + '_sqFlucts.npz', sqFlucts=sqFlucts, bf=bfactors, k=k, cc=coeff, nModes=nModes)
    ax.plot(np.arange(bfactors.shape[0])[:int(n_asym)], bfactors[:int(n_asym)], label='B-factors')
    ax.plot(np.arange(sqFlucts.shape[0])[:int(n_asym)], sqFlucts[:int(n_asym)], label='Squared Fluctuations')
    ax.set_ylabel(r'$Å^{2}$', fontsize=12)
    ax.set_xlabel('Residue Number', fontsize=12)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)

    ax.legend()
    fig.suptitle(
        'Squared Fluctuations vs B-factors: ' + title.title() + ' (' + pdb + ')' + "\n" + r' $\gamma = $' + "{:.5f}".format(
            gamma) +  r'$\pm$' + "{:.5f}".format(ci) + r' $k_{b}T/Å^{2}$' + '  CC = ' + "{:.5f}".format(coeff), fontsize=12)
    # fig.suptitle('# Modes: ' + str(nModes) + ' Corr. Coeff: ' + str(coeff) + ' Spring Constant: ' + str(gamma), fontsize=16)
    # fig.tight_layout()
    plt.savefig('../results/subdivisions/' + pdb + '_sqFlucts.svg')
    plt.savefig('../results/subdivisions/' + pdb + '_sqFlucts.png')
    plt.show()
    return nModes, gamma


def distanceFlucts(evals, evecs, kirch, n_modes, title):
    from scipy import sparse
    from settings import model
    n_modes = int(n_modes)
    n_atoms = kirch.shape[0]
    print(n_modes)
    print('Direct Calculation Method')
    kirch = kirch.tocoo()
    covariance = sparse.lil_matrix((n_atoms, n_atoms))
    df = sparse.lil_matrix((n_atoms, n_atoms))
    if model == 'anm':
        covariance = con_c(evals[:n_modes].copy(), evecs[:, :n_modes].copy(), covariance, kirch.row, kirch.col)
        covariance = covariance.tocsr()
    else:
        covariance = gCon_c(evals[:n_modes].copy(), evecs[:, :n_modes].copy(), covariance, kirch.row, kirch.col)
        covariance = covariance.tocsr()
        print(covariance.min())

    d = con_d(covariance, df, kirch.row, kirch.col)
    d = d.tocsr()
    d.eliminate_zeros()
    print(d.min())
    # print('Average Fluctuations Between Elements', d.data.mean())
    fluctPlot(d, title)
    return d


def fluctPlot(d, title):
    import matplotlib.pyplot as plt
    print('Plotting Fluctuation Histogram')
    import matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 11}
    matplotlib.rc('font', **font)
    print('Plotting')
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlabel('r$A^{2}$', fontsize=12)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.legend()
    fig.suptitle(
        'Histogram Of Pairwise Fluctuations: ' + title + ' (' + pdb + ')', fontsize=12)

    ax.hist(d.data, bins='fd', histtype='stepfilled', density=True)
    fig.tight_layout()
    plt.show()


def fluctToSims(d, pdb):
    d = d.tocsr()
    d.eliminate_zeros()
    nnDistFlucts = np.mean(np.sqrt(d.data))
    print(nnDistFlucts)
    sigma = 1 / (2 * nnDistFlucts ** 2)
    sims = -sigma * d ** 2
    data = sims.data
    data = np.exp(data)
    sims.data = data
    sparse.save_npz('../results/models/' + pdb + 'sims.npz', sims)

    return sims


@nb.njit(parallel=True)
def cov(evals, evecs, i, j):
    # Calculates the covariance between two residues in ANM. Takes the trace of block so no anisotropy info.
    # Commented section would compute normalized covariances
    n_e = evals.shape[0]
    # n_d = evecs.shape[1]
    tr1 = 0
    # tr2 = 0
    # tr3 = 0
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
    # Calculates the covariance between two residues in GNM
    n_e = evals.shape[0]
    n_d = evecs.shape[1]
    cov = 0
    for n in nb.prange(n_e):
        l = evals[n]
        cov += 1 / l * (evecs[i, n] * evecs[j, n])
    return cov


# @nb.njit()
def gCon_c(evals, evecs, c, row, col):
    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        c[i, j] = gCov(evals, evecs, i, j)
    return c


def con_c(evals, evecs, c, row, col):
    # from pythranFuncs import cov
    n_d = int(evecs.shape[0] / 3)
    n_e = evals.shape[0]

    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        c[i, j] = cov(evals, evecs, i, j)
    return c


# @nb.njit()
def con_d(c, d, row, col):
    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        d[i, j] = np.abs(c[i, i] + c[j, j] - 2 * c[i, j])
    return d
