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
        capsid, calphas, title, header, masses, bfactors = getPDB(pdb)
        coords = calphas.getCoords()

        bfactors = calphas.getBetas()

        print('bf: ', bfactors.shape)
        #bfactors = np.tile(bfactors, 60)
        print('bf: ', bfactors.shape)

    # m = np.sum(capsid.getMasses()) / coords.shape[0]
    print(title)

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
        evals, evecs = modeCalc(pdb, hess, kirch, n_modes, eigmethod, model, masses)

    evPlot(evals, evecs, title)
    nm, gamma = mechanicalProperties(bfactors, evals, evecs, title, coords, calphas, 1)

    distFlucts = distanceFlucts(evals, evecs, kirch, n_modes, title, coords)
    sims = fluctToSims(distFlucts, pdb)

    # return -1


def getPDB(pdb):
    from settings import pdbx, local, chains, chains_clust

    dir = '../data/capsid_pdbs/'

    if pdbx:
        filename = dir + pdb + '_full.cif'
    else:
        filename = dir + pdb + '_full.pdb'

    if os.path.exists(filename) or local:
        pass
    else:
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
        asym = parsePDB(filename)
        print(capsid)
    if type(capsid) is list:
        capsid = capsid[0]
    ENM_capsid = capsid.select('protein').copy()

    from ENM import buildMassesCoords

    bfactors, masses = buildMassesCoords(asym.select('protein').copy())

    print(masses.shape)

    if chains:
        ENM_capsid = ENM_capsid.select(chains)


    ENM_capsid = addNodeID(ENM_capsid)

    calphas = ENM_capsid.select('protein and name CA')
    print('Number Of Residues: ', calphas.getCoords().shape[0])


    clustCapsid = capsid.select('protein')
    if chains_clust:
        clustCapsid = clustCapsid.select(chains_clust)

    if not os.path.exists(dir + pdb + '_ca.pdb'):
        print('Writing calphas PDB')
        writePDB(dir + pdb + '_ca.pdb', calphas, hybrid36=True)
    if not os.path.exists(dir + pdb + '_capsid.pdb'):
        print('Writing complete capsid PDB')
        writePDB(dir + pdb + '_capsid.pdb', clustCapsid, hybrid36=True)

    if 'title' in header:
        title = header['title']
    else:
        title = pdb

    return capsid, calphas, title, header, masses, bfactors

def getPDBx(pdb):
    import biotite.database.rcsb as rcsb
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io as strucio
    file_name = rcsb.fetch(pdb, "pdbx", target_path="../data/capsid_pdbs")
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(file_name)
    capsid = pdbx.get_assembly(pdbx_file, assembly_id="1", model=1, extra_fields=['b_factor'])
    title = pdb #pdbx_file.get_category('pdbx_database_related')['details']
    print("Number of protein chains:", struc.get_chain_count(capsid))
    #capsid = capsid[capsid.filter_amino_acids()].copy()
    #strucio.save_structure( '../data/capsid_pdbs/'  + pdb + '_full.cif', capsid)
    calphas = capsid[capsid.atom_name == 'CA']
    coords = calphas.coord
    bfactors = calphas.b_factor
    return capsid, calphas, coords, bfactors, title


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


def modeCalc(pdb, hess, kirch, n_modes, eigmethod, model, masses):
    # from input import model#, eigmethod
    print('Calculating Normal Modes')
    start = time.time()

    useMass=True
    if useMass:
        from scipy.sparse import diags
        mass = np.tile(np.repeat(masses,3), 60)
        print('mass variance: ', np.std(mass))
        print('mass mean: ', np.mean(mass))
        print('mass min: ', np.min(mass))
        print('mass max: ', np.max(mass))
        Mass = diags(mass)

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
    n_dim = mat.shape[0]
    if eigmethod == 'eigsh':
        if useMass:
            evals, evecs = eigsh(mat, k=n_modes, M=Mass, sigma=1e-10, which='LA')
        else:
            evals, evecs = eigsh(mat, k=n_modes, sigma=1e-10, which='LA')
    elif eigmethod == 'lobpcg':
        from scipy.sparse.linalg import lobpcg
        from pyamg import smoothed_aggregation_solver
        diag_shift = 1e-5 * sparse.eye(mat.shape[0])
        mat += diag_shift
        ml = smoothed_aggregation_solver(mat)
        mat -= diag_shift
        M = ml.aspreconditioner()
        epredict = np.random.rand(n_dim, n_modes + 6)
        evals, evecs = lobpcg(mat, epredict, M=M, largest=False, tol=0, maxiter=n_dim)
        evals = evals[6:]
        evecs = evecs[:, 6:]
    elif eigmethod == 'lobcuda':
        import cupy as cp
        from cupyx.scipy.sparse.linalg import lobpcg as clobpcg
        from cupyx.scipy.sparse.linalg import LinearOperator, spilu

        sparse_gpu = cp.sparse.csr_matrix(mat.astype(cp.float32))
        epredict = cp.random.rand(n_dim, n_modes + 6, dtype = cp.float32)

        lu = spilu(sparse_gpu, fill_factor=50)  # LU decomposition
        M = LinearOperator(mat.shape, lu.solve)
        print('gpu eigen')

        evals, evecs = clobpcg(sparse_gpu, epredict, M=M,  largest=False, tol=0, verbosityLevel=0)
        if model=='anm':
            evals = cp.asnumpy(evals[6:])
            evecs = cp.asnumpy(evecs[:, 6:])
        else:
            evals = cp.asnumpy(evals[1:])
            evecs = cp.asnumpy(evecs[:, 1:])
    elif eigmethod == 'eigshcuda':
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import cupyx.scipy.sparse.linalg as cpsp_la
        sigma = 1e-10
        sparse_gpu_shifted = cp.sparse.csr_matrix((mat - sigma*sparse.eye(mat.shape[0])).astype(cp.float32))
        #mat_shifted = sparse.csc_matrix(mat - sigma*sparse.eye(mat.shape[0]))
        #lu = sparse.linalg.splu(mat_shifted)
        A_gpu_LU = cpsp_la.splu(sparse_gpu_shifted)  # LU decomposition
        #A_gpu_LO = cpsp_la.LinearOperator(mat_shifted.shape, lu.solve)  # Linear Operator
        A_gpu_LO = cpsp_la.LinearOperator(sparse_gpu_shifted.shape, A_gpu_LU.solve)

        eigenvalues_gpu, eigenstates_gpu = cpsp_la.eigsh(A_gpu_LO, k=n_modes, which='LA', tol=0)

        eigenvalues_gpu = eigenvalues_gpu.get()
        eigenstates_gpu = eigenstates_gpu.get()
        eigenvalues_gpu = (1 + eigenvalues_gpu * sigma) / eigenvalues_gpu
        idx = np.argsort(eigenvalues_gpu)
        eigenvalues_gpu = eigenvalues_gpu[idx]
        evals = cp.asnumpy(eigenvalues_gpu)
        evecs = cp.asnumpy(eigenstates_gpu)
    if cuth_mkee:
        evecs = evecs[perm, :].copy()
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
        'Eigenvalues/Squared Frequencies: ' + ' (' + pdb + ')', fontsize=12)

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


def mechanicalProperties(bfactors, evals, evecs, title, coords, calphas, m):
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
    nModes, coeff, k, sqFlucts, stderr, r2 = fluctFit(evals, evecs, bfactors)



    boltz = 1.380649e-23
    dalt = 6.022173643 * 10 ** 26
    bz = boltz * 10 ** 20 * dalt
    T_scale = 270
    T_sim = 270

    gamma = T_scale * bz * (8 * np.pi ** 2) / k

    if model == 'anm':
        gamma = gamma / 3

    gamma = gamma * m  # average mass in daltons per residue
    gcgs = 1.66054e-24
    gamma_cgs = gamma*gcgs
    print(gamma_cgs)
    scale = T_sim * bz

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 11}
    matplotlib.rc('font', **font)

    stderr =  stderr/k
    stderr = gamma_cgs * stderr
    from bfactorFit import confidenceInterval
    ci = confidenceInterval(bfactors, stderr)

    print(nModes, coeff, gamma)
    n_asym = int(bfactors.shape[0]/60)
    np.savez('../results/subdivisions/' + pdb + '_sqFlucts.npz', sqFlucts=sqFlucts, bf=bfactors, k=k, cc=coeff, nModes=nModes)
    ax.plot(np.arange(bfactors.shape[0])[:n_asym], bfactors[:n_asym], label='B-factors')
    ax.plot(np.arange(sqFlucts.shape[0])[:n_asym], sqFlucts[:n_asym], label='Squared Fluctuations')
    ax.set_ylabel(r'$Å^{2}$', fontsize=12)
    ax.set_xlabel('Residue Number', fontsize=12)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)

    ax.legend()
    fig.suptitle(
        'Squared Fluctuations vs B-factors: '  + ' (' + pdb + ')' + "\n" + r' $\gamma = $' + "{:.3e}".format(
            gamma_cgs) +  r'$\pm$' + "{:.3e}".format(ci) + r'$ \frac{dyn}{cm}$' + '  CC = ' + "{:.3f}".format(coeff) + '  r2 = ' + "{:.3f}".format(r2), fontsize=12)
    # fig.suptitle('# Modes: ' + str(nModes) + ' Corr. Coeff: ' + str(coeff) + ' Spring Constant: ' + str(gamma), fontsize=16)
    # fig.tight_layout()
    plt.savefig('../results/subdivisions/' + pdb + '_sqFlucts.svg')
    plt.savefig('../results/subdivisions/' + pdb + '_sqFlucts.png')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 11}
    matplotlib.rc('font', **font)

    ax.scatter(sqFlucts, bfactors, label='B-factors')
    ax.set_ylabel(r'$Å^{2}$', fontsize=12)
    ax.set_xlabel('Residue Number', fontsize=12)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)

    ax.legend()

    plt.show()

    if model=='anm':
        from score import volFlucts
        compressibility, vrms, vplot, vmodes = volFlucts(coords, evals, evecs, gamma=gamma)
        vflucts = np.sqrt(scale*vrms)
        vplot = np.sqrt(scale*vplot)
        print('volume fluctuations: ',vflucts, r'$Å^{3}$')

        spa = 1e-10 * 6.022e26
        betaT_pa = spa*compressibility*10e9
        print()


        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        font = {'family': 'sans-serif',
                'weight': 'normal',
                'size': 11}
        matplotlib.rc('font', **font)
        print('Plotting compressibility by modes')
        ax.set_ylabel(r'$\langle \Delta V \rangle Å^{3}$', fontsize=12)
        ax.set_xlabel('# of low frequency modes', fontsize=12)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.legend()
        fig.suptitle(
            'Modes vs Volume Fluctuations: ' + ' (' + pdb + ')' + 'Isothermal Compressibility ' + r'$\beta_T$' + '=' + "{:.5f}".format(betaT_pa) + ' ' r'$GPa^{-1}$', fontsize=12)

        # ax.scatter(np.arange(vplot.shape[0]), scale*vplot, marker='D', s=10, label='eigs')
        ax.bar(np.arange(vplot.shape[0]), vplot)
        ax.set_ylim([0,np.max(vplot)])
        fig.tight_layout()
        plt.savefig('../results/subdivisions/' + pdb + '_compressibility.png')
        plt.show()
        ind = np.argpartition(vplot, -5)[-5:]
        compModes = evecs[:, ind]
        print('Most significant compressibility mode:', np.argmax(vplot))
        # from prody import ANM, writeNMD
        # anm = ANM(pdb)
        # anm._eigvals = evals
        # anm._eigvecs = evecs
        # anm._array = evecs
        # anm._n_modes = evals.shape[0]
        # anm._vars = 1 / evals
        # anm._n_atoms = int(evecs.shape[0]/3)

        #writeNMD(pdb + '_compress.nmd', anm[ind], calphas)

    return nModes, gamma


def distanceFlucts(evals, evecs, kirch, n_modes, title, coords):
    from scipy import sparse
    from settings import model, d_cutoff
    from sklearn.neighbors import BallTree, radius_neighbors_graph

    n_modes = int(n_modes)
    n_atoms = kirch.shape[0]
    print(n_modes)
    print('Direct Calculation Method')

    tree = BallTree(coords)
    adj = radius_neighbors_graph(tree, d_cutoff, mode='connectivity', n_jobs=-1)
    adj = adj.tocoo()

    covariance = sparse.lil_matrix((n_atoms, n_atoms))
    df = sparse.lil_matrix((n_atoms, n_atoms))

    if model == 'anm':
        covariance = con_c(evals[:n_modes].copy(), evecs[:, :n_modes].copy(), covariance, adj.row, adj.col)
        covariance = covariance.tocsr()
    else:
        covariance = gCon_c(evals[:n_modes].copy(), evecs[:, :n_modes].copy(), covariance, adj.row, adj.col)
        covariance = covariance.tocsr()
        print(covariance.min())

    d = con_d(covariance, df, adj.row, adj.col)
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
    cov = tr1#  / np.sqrt(tr2 * tr3)
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


#@nb.njit()
def gCon_c(evals, evecs, c, row, col):
    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        c[i, j] = gCov(evals, evecs, i, j)
    return c

#@nb.njit()
def con_c(evals, evecs, c, row, col):
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
        d[i, j] = np.abs(c[i, i] + c[j, j] - 2 * c[i, j])
    return d
