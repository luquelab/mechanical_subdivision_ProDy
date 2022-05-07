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

    capsid, calphas, title = getPDB(pdb)
    coords = calphas.getCoords()
    print(title)
    global anm, n_atoms, n_dim, n_asym
    n_atoms = calphas.getCoords().shape[0]
    print(n_atoms)
    n_dim = 3*n_atoms
    n_asym = int(n_atoms/60)

    if model=='gnm':
        anm = GNM(pdb + '_full')
        n_dim = n_atoms
    else:
        anm = ANM(pdb + '_full')

    if mode =='full':
        if model == 'gnm':
            kirch = buildKirch(pdb, calphas, cutoff)
            hess = kirch
        else:
            hess, kirch = buildHess(pdb, calphas, cutoff)
    else:
        if model == 'gnm':
            kirch = loadKirch(pdb)
            hess = kirch
        else:
            hess, kirch = loadHess(pdb)



    if mode == 'eigs':
        evals, evecs, kirch = loadModes(pdb, n_modes)
    else:
        evals, evecs = modeCalc(pdb, hess, kirch, n_modes, eigmethod)

    # anm._n_atoms = n_atoms
    # anm._vars = 1/evals
    # print(anm.numAtoms())
    # print(calphas.numAtoms())
    # writeNMD('test.nmd', anm[:20], calphas)



    # evPlot(evals, evecs)
    icoEvPlot(evals, evecs, calphas)
    bfactors = calphas.getBetas()
    n_modes, gamma = mechanicalProperties(bfactors, evals, evecs, coords, hess)

    distFlucts = distanceFlucts(evals, evecs, kirch, n_modes)
    from score import stresses
    rStress = stresses(hess, distFlucts)
    sims = fluctToSims(distFlucts, pdb)
    saveAtoms(calphas, filename='../results/models/' + 'calphas_' + pdb)


    #from eigenCount import eigenCutoff

    #n_modes = eigenCutoff(evals, 0.001)
    #evals = evals[:n_modes]
    return sims, calphas


def getPDB(pdb):
    from input import pdbx
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
        capsid, header = parsePDB(filename, header=True, biomol=True)
    calphas = capsid.select('calpha').copy()
    print('Number Of Residues: ', calphas.getCoords().shape[0])
    os.chdir('../../src')

    writePDB('../results/subdivisions/' + pdb + '_ca.pdb', calphas,
             hybrid36=True)
    title = header['title']

    return capsid, calphas, title


def gammaDist(dist2, *args):
    return 1/dist2

def buildHess(pdb, calphas, cutoff=10.0):

    anm.buildHessian(calphas, cutoff=cutoff, kdtree=True, sparse=True, gamma=gammaDist)
    sparse.save_npz('../results/models/' + pdb + 'hess.npz', anm.getHessian())
    sparse.save_npz('../results/models/' + pdb + 'kirch.npz', anm.getKirchhoff())
    kirch = anm.getKirchhoff()

    return anm.getHessian(), kirch

def buildKirch(pdb, calphas, cutoff=10.0):

    anm.buildKirchhoff(calphas, cutoff=cutoff, kdtree=True, sparse=True, gamma=gammaDist)
    sparse.save_npz('../results/models/' + pdb + 'kirch.npz', anm.getKirchhoff())
    kirch = anm.getKirchhoff()

    return kirch


def loadHess(pdb):
    hess = sparse.load_npz('../results/models/' + pdb + 'hess.npz')
    anm._hessian = hess
    kirch = sparse.load_npz('../results/models/' + pdb + 'kirch.npz')
    return hess, kirch

def loadKirch(pdb):
    kirch = sparse.load_npz('../results/models/' + pdb + 'kirch.npz')
    return kirch


def modeCalc(pdb, hess, kirch, n_modes, method):
    from input import model, eigmethod
    print('Calculating Normal Modes')
    start = time.time()

    if model=='anm':
        mat = hess
    else:
        mat = kirch
    if eigmethod=='eigsh':
        evals, evecs = eigsh(hess, k=n_modes, sigma=1e-8, which='LA')
    elif eigmethod=='lobpcg':
        from scipy.sparse.linalg import lobpcg
        # if not hasattr(np, "float128"):
        #     np.float128 = np.longdouble  # #698
        # diag_shift = 1e-5 * sparse.eye(mat.shape[0])
        # mat += diag_shift
        # ml = smoothed_aggregation_solver(hess)
        # M = ml.aspreconditioner()
        #
        # hess -= diag_shift
        epredict = np.random.rand(n_dim, n_modes+6)
        evals, evecs = lobpcg(hess, epredict, largest=False, tol=0, maxiter=n_dim)
        evals = evals[6:]
        evecs = evecs[:,6:]
        print(evecs.shape)
    elif eigmethod=='lobcuda':
        import cupy as cp
        from cupyx.scipy.sparse.linalg import lobpcg as clobpcg
        sparse_gpu = cp.sparse.csr_matrix(hess.astype(cp.float32))
        epredict = cp.random.rand(n_dim, n_modes+6)
        evals, evecs = clobpcg(sparse_gpu, epredict, largest=False, tol=0, maxiter=n_dim)
        evals = cp.asnumpy(evals[6:])
        evecs = cp.asnumpy(evecs[:, 6:])
        print(evecs.shape)
    print(evals)
    end = time.time()
    print(end - start)
    print(evals[:6])
    np.savez('../results/models/' + pdb + model + 'modes.npz', evals=evals, evecs=evecs)
    return evals, evecs


def loadModes(pdb, n_modes):
    from input import model
    if model=='anm':
        modes = np.load('../results/models/' + pdb + model + 'modes.npz')
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


def evPlot(evals, evecs):
    from prody import writeNMD
    import matplotlib.pyplot as plt
    print('Plotting')
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.scatter(np.arange(evals.shape[0]), evals, marker='D', label='eigs')
    #ax[1].plot(np.arange(evecs.shape[1]), evecs[0,:], label='1st Mode')
    #ax[1].plot(np.arange(evecs.shape[1]), evecs[6, :], label='60th Mode')
    fig.tight_layout()
    plt.show()

def icoEvPlot(evals, evecs, calphas):
    import matplotlib.pyplot as plt
    from input import pdb
    uniques, inds, counts = np.unique(evals.round(decimals=6), return_index=True, return_counts=True)
    icoEvalInds = inds[counts==1]
    print(icoEvalInds)
    icoEvals = evals[icoEvalInds]
    icoEvecs = evecs[:,icoEvalInds]
    anm._eigvals = evals
    anm._eigvecs = evecs
    anm._array = evecs
    anm._n_modes = evals.shape[0]
    anm._vars = 1/evals
    print(icoEvecs.shape)
    print(icoEvals)
    anm._n_atoms = calphas.getCoords().shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.scatter(np.arange(icoEvals.shape[0]), icoEvals, marker='D', label='eigs')
    plt.show()
    from prody import sliceModel
    anm_save, calphas_save = sliceModel(anm, calphas, calphas[:n_asym])
    writeNMD(pdb + '_ico.nmd', anm[icoEvalInds], calphas)


def mechanicalProperties(bfactors, evals, evecs, coords, hess):
    from input import pdb
    import matplotlib
    import matplotlib.pyplot as plt
    from optcutoff import fluctFit
    from score import collectivity, meanCollect, effectiveSpringConstant, overlapStiffness, globalPressure
    _, _, title = getPDB(pdb)
    print('Plotting')
    nModes, coeff, k, sqFlucts = fluctFit(evals, evecs, bfactors)

    #compressibility, stiffneses = effectiveSpringConstant(coords, evals[:nModes], evecs[:,:nModes])
    from score import globalPressure
    # bulkmod = globalPressure(coords, hess, 1)
    # print('Bulk Modulus 1: ', bulkmod)
    # print('Bulk Modulus 2: ', 1 / compressibility)
    # plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 11}
    matplotlib.rc('font', **font)
    # kb = 1.38065 * 10**-23
    # T = 293
    # da = 110*1.66*10**-27
    # angs = 10^20
    # scaledKb = T*kb*da*angs
    gamma = (8 *np.pi**2)/k

    print(nModes, coeff, gamma)
    kc = collectivity(sqFlucts)
    kcmean = meanCollect(evecs, evals, bfactors)
    from score import meanStiff
    mstiff = meanStiff(evals)
    np.savez('../results/subdivisions/' + pdb + '_sqFlucts.npz', sqFlucts=sqFlucts, k=k, cc=coeff, nModes=nModes)
    ax.plot(np.arange(bfactors.shape[0])[:int(n_asym)], bfactors[:int(n_asym)], label='B-factors')
    ax.plot(np.arange(sqFlucts.shape[0])[:int(n_asym)], sqFlucts[:int(n_asym)], label='Squared Fluctuations')
    ax.set_ylabel(r'$Å^{2}$', fontsize=12)
    ax.set_xlabel('Residue Number', fontsize=12)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)

    ax.legend()
    fig.suptitle('Squared Fluctuations vs B-factors: ' + title.title() + ' (' + pdb + ')' + "\n" + r' $\gamma = $' + "{:.5f}".format(gamma) + r' $k_{b}T/Å^{2}$' + '  CC = ' +"{:.5f}".format(coeff) ,fontsize=12)
    # fig.suptitle('# Modes: ' + str(nModes) + ' Corr. Coeff: ' + str(coeff) + ' Spring Constant: ' + str(gamma), fontsize=16)
    # fig.tight_layout()
    plt.savefig('../results/subdivisions/' + pdb + '_sqFlucts.svg')
    plt.show()
    return nModes, gamma


def distanceFlucts(evals, evecs, kirch, n_modes):
    from scipy import sparse
    from input import model
    n_modes = int(n_modes)
    n_atoms = kirch.shape[0]
    print(n_modes)
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

    d = con_d(covariance, df, kirch.row, kirch.col)
    d = d.tocsr()
    d.eliminate_zeros()
    print(d.min())
    # print('Average Fluctuations Between Elements', d.data.mean())
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

# def covDiags(evals, evecs):
#     n_atoms = int(evecs.shape[0]/3)
#     n_evals = evals.shape[0]
#     covDiags = np.zeros((n,3,3))
#     for i in range(n_evals):
#         ev = evals[i]
#         for j in range(n_atoms):
#             vec = evecs[3*j:3*j+3]
#             block = 1/ev * np.outer(vec,vec)
#             covDiags[j] += block
#     return covDiags

def sphericalTransform(block, coord):
    x = coord[0]
    y = coord[1]
    z = coord[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y,x)
    theta = np.arccos(z,r)
    j1 = np.cos(phi) * np.sin(theta)
    j2 = -r*np.sin(phi) * np.sin(theta)
    j3 = r*np.cos(phi) * np.cos(theta)
    j4 = np.sin(phi) * np.sin(theta)
    j5 = r*np.cos(phi) * np.sin(theta)
    j6 = r*np.sin(phi) * np.cos(theta)
    j7 = np.cos(theta)
    j8 = 0
    j9 = -r*np.cos(theta)
    jac = np.array([[j1,j2,j3],[j4,j5,j6],[j7,j8,j9]])
    sphereBlock = jac.T @ block @ jac
    return sphereBlock



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
    # from pythranFuncs import cov
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