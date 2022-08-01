import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from settings import *

def buildENM(calphas, coords, bfactors):
    from scipy import sparse
    import numpy as np
    from sklearn.neighbors import BallTree, radius_neighbors_graph, kneighbors_graph
    from settings import cbeta

    n_atoms = coords.shape[0]
    n_asym = int(n_atoms/60)

    print('# Atoms ',n_atoms)
    dof = n_atoms * 3

    tree = BallTree(coords)
    kirch = radius_neighbors_graph(tree, cutoff, mode='distance', n_jobs=-1)
    kc = kirch.tocoo().copy()
    kc.sum_duplicates()
    kirch = kirchGamma(kc, bfactors, d2=d2, flexibilities=flexibilities, struct=False)
    backbone = True
    if backbone:
        kbb = 5
        kirch = backbonePrody(calphas, kirch.tolil(), kbb, s=3)
    print('nonzero values: ', kirch.nnz)
    dg = np.array(kirch.sum(axis=0))
    kirch.setdiag(-dg[0])
    kirch.sum_duplicates()
    kirch = kirch.tocsr()
    #print(kirch.data)
    print('kirch: ', kirch.diagonal(k=0))

    if model=='anm':
        kc = kirch.tocoo().copy()
        kc.sum_duplicates()
        hData = hessCalc(kc.row, kc.col, kirch.data, coords)
        indpt = kirch.indptr
        inds = kirch.indices
        hessian = sparse.bsr_matrix((hData, inds, indpt), shape=(dof,dof)).tocsr()
        hessian = fanm*hessian + (1-fanm)*sparse.kron(kirch, np.identity(3))
    else:
        hessian = kirch.copy()
    print('done constructing matrix')
    # fig, ax = plt.subplots()
    # mat = ax.matshow(hessian[:int(3 * n_asym / 10), :int(3 * n_asym / 10)].todense())
    # plt.colorbar(mat, ax=ax)
    # plt.show()
    return kirch, hessian


def buildMassesCoords(atoms):
    print(atoms[0])
    coords = []
    masses = []
    print('segments:', atoms.numSegments())
    for seg in atoms.iterSegments():
        for chain in seg.iterChains():
            for res in chain.iterResidues():
                mass = np.sum(res.getMasses())
                masses.append(mass)
                # if res.getResname() == 'GLY':
                #     coord = np.mean(res.getCoords())
                # else:
                #     coord = res['CA'].getCoords()
                coord = res['CA'].getCoords()
                coords.append(coord)

    return np.asarray(coords), np.asarray(masses)

def kirchGamma(kirch, bfactors, **kwargs):
    kg = kirch.copy()

    # if 'struct' in kwargs and kwargs['struct']:
    #     chains = atoms.getData('chainNum')
    #     print(chains.shape)
    #     #sstr, ssid, chainNum = tup
    #     print('# secstr residues:', np.count_nonzero(atoms.getSecstrs() != 'C'))
    #     tup = (atoms.getSecstrs(), atoms.getSecids(), chains)
    #     abg = cooOperation(kirch.row, kirch.col, kirch.data, secStrGamma, tup)
    #     kg.data = abg
    #     # kg = addSulfideBonds(sulfs, kg)
    if 'd2' in kwargs and kwargs['d2']:
        print('d2 mode')
        kg.data = -1/((kg.data)**2)
    else:
        kg.data = -np.ones_like(kg.data)

    if 'flexibilities' in kwargs and kwargs['flexibilities']:
        flex = flexes(1/bfactors)
        fl = cooOperation(kirch.row, kirch.col, kirch.data, flexFunc, flex)
        kg.data = kg.data*fl

    # if 'cbeta' in kwargs and kwargs['cbeta']:
    #     names = atoms.getNames()
    #     abgamma = np.array([0 if x == 'CA' else np.sqrt(0.5) for x in names])
    #     abg = cooOperation(kirch.row, kirch.col, kirch.data, abFunc, abgamma)
    #     kg.data = kg.data * abg
    return kg

# @nb.njit()
# def backboneKirch(chids, asymids, resids):
#
#     for i in range(len(chids)):
#         for j in range(len(chids)):
#             sij = np.abs(resids[i] - resids[j])
#             if sij == 1 and chids[i] == chids[j] and asymids[i] == asymids[j]:
#
#             elif sij <= 3 and chids[i] == chids[j] and asymids[i] == asymids[j]:
#                 return -1

@nb.njit()
def cooOperation(row, col, data, func, arg):
    r = np.copy(data)
    for n, (i, j, v) in enumerate(zip(row, col, data)):
        if i==j:
            continue
        r[n] = func(i,j,v, arg)
    return r

@nb.njit()
def abFunc(i, j, d, abg):
    return -abg[i]*abg[j] + 1.0

def flexes(bfactors):
    return bfactors/np.mean(bfactors)

@nb.njit()
def d2Func(d2):
    return 1/d2

@nb.njit()
def flexFunc(i,j,d, fl):
    return np.sqrt(fl[i]*fl[j])

@nb.njit()
def structFunc(i,j,d, chids, asymids):
    sij = np.abs(i-j)
    if sij == 1 and chids[i]==chids[j] and asymids[i]==asymids[j]:
        return -100
    elif sij <= 3 and chids[i]==chids[j] and asymids[i]==asymids[j]:
        return -1
    else:
        return -1/(d)**2

@nb.njit()
def secStrGamma(i, j, d, tup):
    """Returns force constant."""
    sstr, ssid, chainNum = tup
    if d <= 4 and chainNum[i]==chainNum[j]:
        return -10
    # if residues are in the same secondary structure element
    if ssid[i] == ssid[j] and chainNum[i]==chainNum[j]:
        i_j = abs(i - j)
        if ((i_j <= 4 and sstr[i] == 'H') or
            (i_j <= 3 and sstr[i] == 'G') or
            (i_j <= 5 and sstr[i] == 'I')) and d <= 7:
            return -6.
    elif sstr[i] == sstr[j] == 'E' and d <= 6:
        return -6.

    return -1/d**2


@nb.njit()
def hessCalc(row, col, kGamma, coords):
    hData = np.zeros((row.shape[0], 3, 3))
    dinds = np.nonzero(row==col)[0]
    for k, (i,j) in enumerate(zip(row,col)):
        if i==j:
            continue
        dvec = coords[j] - coords[i]
        d2 = np.dot(dvec, dvec)
        g = kGamma[k]
        hblock = np.outer(dvec, dvec) * (g / d2) # + (1-fanm)*np.identity(3)*g
        hData[k,:,:] = hblock
        hData[dinds[i]] += -hblock/2
        hData[dinds[j]] += -hblock/2
    return hData

def backbonePrody(calphas, kirch, k, s):
    nr = calphas.numAtoms()
    count = 0
    chid = 0
    ch0 = calphas[0].getChid()
    seg0 = calphas[0].getSegname()
    # kirch[0,1:3] = -k
    # kirch[1:3, 0] = -k
    nch = 0
    kbbs = np.array([k,k/2,k/4])[:s]
    for i, ca in enumerate(calphas.iterAtoms()):
        if count == 0:
            kirch[i, (i + 1):(i + 1 + s)] = -kbbs
            kirch[(i + 1):(i + 1 + s), i] = -kbbs
            count += 1
            continue
        elif count<=s:
            kbbs = kbbs[::-1]
            kirch[i, (i - count):i] = -kbbs[-count:]
            kirch[(i - count):i, i] = -kbbs[-count:]
            count += 1
            continue
        if ca.getChid() == ch0 and ca.getSegname() == seg0:
            kirch[i, (i-s):i] = -kbbs
            kirch[(i - s):i,i] = -kbbs
            count += 1
        else:
            chid += 1
            #print(ch0, seg0, 'done')
            ch0 = ca.getChid()
            seg0 = ca.getSegname()
            count = 0
            nch += 1
    print(nch)
    return kirch.tocsr()



def betaTestKirch(coords):
    from sklearn.neighbors import BallTree, radius_neighbors_graph
    from settings import cutoff
    tree = BallTree(coords)
    kirch = radius_neighbors_graph(tree, cutoff, mode='distance', n_jobs=-1)
    abk, betaKirch, k = betaCarbonModel(kirch, coords)

    rowstops = [0]
    stop = 0
    for i, row in enumerate(abk[0]):
        rowlen = row.shape[0]
        stop += rowlen
        rowstops.append(stop)
    inds = np.hstack(abk[0].flatten())
    data = np.hstack(abk[1].flatten())
    rowstops = np.array(rowstops)
    print(rowstops.shape)

    from scipy import sparse
    abKirch = sparse.csr_matrix((data, inds, rowstops), shape=k.shape)
    abKirch.setdiag(0)

    betaKirch[0, :] = 0
    betaKirch[:, 0] = 0
    betaKirch[-1, :] = 0
    betaKirch[:, -1] = 0

    abKirch[0, :] = 0
    abKirch[:, 0] = 0
    abKirch[-1, :] = 0
    abKirch[:, -1] = 0
    abKirch.eliminate_zeros()
    betaKirch.eliminate_zeros()


    abKirch.data = -1/abKirch.data**2
    betaKirch.data = -1 / betaKirch.data ** 2
    kirch.data = -1 / kirch.data ** 2
    fullKirch = 1 * kirch + 0.5 * betaKirch + 0.5 * abKirch
    fullKirch.eliminate_zeros()

    return fullKirch

def betaCarbonModel(calphas):
    from settings import cutoff, d2, fanm
    from sklearn.neighbors import BallTree, radius_neighbors_graph
    from scipy import sparse
    coords = calphas.getCoords()
    n_atoms = coords.shape[0]
    n_asym = int(n_atoms / 60)

    print('# Atoms ', n_atoms)
    dof = n_atoms * 3
    betaCoords, bvals, nobetas = buildBetas(coords, calphas)
    print(bvals.shape)
    print(betaCoords.shape)

    tree = BallTree(coords)
    kirch = radius_neighbors_graph(tree, cutoff, mode='distance', n_jobs=-1)
    kc = kirch.tocoo().copy()
    kc.sum_duplicates()
    bfactors = calphas.getBetas()
    kirch = kirchGamma(kc, bfactors, d2=d2)
    backbone = True
    if backbone:
        kbb = 5
        kirch = backbonePrody(calphas, kirch.tolil(), kbb, s=1)
    print('nonzero values: ', kirch.nnz)
    dg = np.array(kirch.sum(axis=0))
    kirch.setdiag(-dg[0])
    kirch.sum_duplicates()
    kirch = kirch.tocsr()

    btree = BallTree(betaCoords)
    betaKirch = radius_neighbors_graph(btree, cutoff, mode='distance',  n_jobs=-1)
    betaKirch = kirchGamma(betaKirch.tocoo(), bfactors, d2=d2).tocsr()
    #abKirch = tree.query_radius(betaCoords, r=cutoff, return_distance=True).tocoo()

    if model=='anm':
        kc = kirch.tocoo().copy()
        kc.sum_duplicates()
        haaData = hessCalc(kc.row, kc.col, kirch.data, coords)
        indpt = kirch.indptr
        inds = kirch.indices
        haa = sparse.bsr_matrix((haaData, inds, indpt), shape=(dof, dof)).tolil()
        haa = fanm * haa + (1 - fanm) * sparse.kron(kirch, np.identity(3))
        haa = haa.tocsr()
        haa.eliminate_zeros()
        kbc = betaKirch.tocoo().copy()
        kbc.sum_duplicates()
        row, col = (kbc.row, kbc.col)
        hbb = bbHessCalcSlow(row, col, betaCoords, bvals, nobetas, kbc.data).tocsr()
        hbb = fanm * hbb + (1 - fanm) * sparse.kron(kbc, np.identity(3))
        # ijr, hbbData = bbHessCalc(row, col, betaCoords, bvals, nobetas, kbc.data)
        # ijr = np.array(ijr)
        # hbbData = np.array(hbbData)
        # hbb = sparse.coo_matrix((hbbData, (ijr[:,0],ijr[:,1])), shape=(dof, dof)).tocsr()
        hbb.eliminate_zeros()
        hbb.sum_duplicates()

        # row, col = (abKirch.row, abKirch.col)
        # hab = abHessCalcSlow(row, col, betaCoords, coords, bvals, nobetas, abKirch.data).tocsr()
        # # ijr, hbbData = bbHessCalc(row, col, betaCoords, bvals, nobetas, kbc.data)
        # # ijr = np.array(ijr)
        # # hbbData = np.array(hbbData)
        # # hbb = sparse.coo_matrix((hbbData, (ijr[:,0],ijr[:,1])), shape=(dof, dof)).tocsr()
        # hab.eliminate_zeros()
        # hab.sum_duplicates()

        from sklearn.utils.validation import check_symmetric
        check_symmetric(haa, raise_exception=True, tol=1e-5)
        check_symmetric(hbb, raise_exception=True, tol=1e-5)
        #check_symmetric(hab, raise_warning=True, tol=1e-5)

    hess = haa + hbb/2 #+ hab/2
    print(hess.data)
    kirch = betaKirch + kirch
    hess.eliminate_zeros()
    fig, ax = plt.subplots()
    mat = ax.matshow(hess[:int(3*n_asym), :int(3*n_asym)].todense())
    plt.colorbar(mat, ax=ax)
    plt.show()

    fig, ax = plt.subplots()
    mat = ax.matshow(haa[:int(3 * n_asym), :int(3 * n_asym)].todense())
    plt.colorbar(mat, ax=ax)
    plt.show()

    return kirch, hess


# @nb.njit()
def buildBetas(coords, calphas):
    na = calphas.numAtoms()
    noBetas = []
    bvals = []
    bcoords = np.zeros_like(coords)
    count = 0
    chid = 0
    ch0 = calphas[0].getChid()
    seg0 = calphas[0].getSegname()
    nch = 0
    for i, ca in enumerate(calphas.iterAtoms()):
        if count == 0 or ca.getResname()=='GLY' or i==na-1:
            bcoords[i, :] = coords[i, :]
            bvals.append(1)
            noBetas.append(i)
            count += 1
        elif ca.getChid() == ch0 and ca.getSegname() == seg0:

            r = 2 * coords[i, :] - coords[i - 1, :] - coords[i + 1, :]
            b = 1/np.linalg.norm(r)
            bvals.append(b)
            bcoords[i, :] = coords[i, :] + 3.0 * r * b
            count += 1
        else:
            bcoords[i-1, :] = coords[i-1, :]
            bcoords[i, :] = coords[i, :]
            noBetas.append(i-1)
            bvals.append(1)
            chid += 1
            ch0 = ca.getChid()
            seg0 = ca.getSegname()
            count = 0
            nch += 1

    noBetas.append(i)

    return bcoords, np.array(bvals), np.array(noBetas)



@nb.njit()
def betaBetaBond(rvec, bi, bj):
    r2 = np.sum(rvec**2)
    mat = np.outer(rvec, rvec)/r2
    rmat1 = np.zeros((9,9))
    m1 = -9*bi*bj * mat
    rmat1[0:3, 0:3] = m1
    rmat1[0:3, 6:9] = m1
    rmat1[6:9, 0:3] = m1
    rmat1[6:9, 6:9] = m1

    m2 = 3 * bi * (1+6*bj) * mat
    rmat1[3:6, 0:3] = m2
    rmat1[3:6, 6:9] = m2

    m3 = 3 * bj * (1 + 6 * bi) * mat
    rmat1[0:3, 3:6] = m3
    rmat1[6:9, 3:6] = m3

    m4 = -(1+6*bi) * (1 + 6 * bj) * mat
    rmat1[3:6, 3:6] = m4
    #
    #
    # rmat2 = np.zeros((9, 9))
    # m1 = 9 * bi * bi * mat
    # rmat2[0:3, 0:3] = m1
    # rmat2[0:3, 6:9] = m1
    # rmat2[6:9, 0:3] = m1
    # rmat2[6:9, 6:9] = m1
    #
    # m2 = -3 * bi * (1 + 6 * bi) * mat
    # rmat2[3:6, 0:3] = m2
    # rmat2[3:6, 6:9] = m2
    #
    # m3 = -3 * bi * (1 + 6 * bi) * mat
    # rmat2[0:3, 3:6] = m3
    # rmat2[6:9, 3:6] = m3
    #
    # m4 = (1 + 6 * bi) * (1 + 6 * bi) * mat
    # rmat1[3:6, 3:6] = m4

    return rmat1

@nb.njit()
def alphaBetaBond(rvec, bj):

    r2 = np.sum(rvec**2)
    mat = np.outer(rvec, rvec)/r2
    rmat1 = np.zeros((3,9))
    rmat2 = np.zeros((9,9))

    m1 = 3*bj * mat
    rmat1[:, 0:3] = m1
    rmat1[:, 6:9] = m1

    m2 = - (1+6*bj) * mat
    rmat1[3:6, 0:3] = m2
    rmat1[3:6, 6:9] = m2

    m1 = 9 * bj * bj * mat
    rmat2[0:3, 0:3] = m1
    rmat2[0:3, 6:9] = m1
    rmat2[6:9, 0:3] = m1
    rmat2[6:9, 6:9] = m1

    m2 = -3 * bj * (1 + 6 * bj) * mat
    rmat2[3:6, 0:3] = m2
    rmat2[3:6, 6:9] = m2
    rmat2[0:3, 3:6] = m2
    rmat2[6:9, 3:6] = m2

    m3 = (1 + 6 * bj) * (1 + 6 * bj) * mat
    rmat2[3:6, 3:6] = m3

    return rmat1, rmat2


@nb.njit()
def bbHessCalc(row, col, bcoords, bvals, nobetas, kGamma):
    hData = [] #np.zeros((6*row.shape[0], 3, 3))
    ijr = [] #np.zeros((6*row.shape[0],2))
    for k, (i,j) in enumerate(zip(row, col)):
        if abs(i-j) > 2 or np.any(nobetas==i) or np.any(nobetas==j):
            continue
        bi = bvals[i]
        bj = bvals[j]
        rvec = bcoords[j] - bcoords[i]
        g = kGamma[k]
        hblock = betaBetaBond(rvec, bi, bj)

        for l in range(9):
            for m in range(9):
                lm = (3*(i-1)+l, 3*(j-1)+m)
                ijr.append(lm)
                hData.append(hblock[l,m])
                ijr.append(lm[::-1])
                hData.append(hblock[l, m])

                ijr.append((3*(i-1)+l, 3*(i-1)+m))
                hData.append(-hblock[l,m]/2)

                ijr.append((3*(j-1)+l, 3*(j-1)+m))
                hData.append(-hblock[l,m] / 2)

    return ijr, hData

def bbHessCalcSlow(row, col, bcoords, bvals, nobetas, kGamma):
    from scipy import sparse
    dof = 3*bcoords.shape[0]
    hbb = sparse.lil_matrix((dof, dof))
    for k, (i,j) in enumerate(zip(row, col)):
        if abs(i-j) <= 1 or np.any(nobetas==i) or np.any(nobetas==j):
            continue
        bi = bvals[i]
        bj = bvals[j]
        rvec = bcoords[j] - bcoords[i]
        g = -kGamma[k]

        ijblock = g*betaBetaBond(rvec, bi, bj)
        jiblock = g*betaBetaBond(rvec, bj, bi)
        iiblock = -g*betaBetaBond(rvec, bi, bi)
        jjblock = -g*betaBetaBond(rvec, bj, bj)
        hbb[3*i-3:3*i+6, 3*j-3:3*j+6] += ijblock
        hbb[3 * j - 3:3 * j + 6, 3 * i - 3:3 * i + 6] += jiblock
        hbb[3*i-3:3*i+6, 3*i-3:3*i+6] += iiblock/2
        hbb[3 * j - 3:3 * j + 6, 3 * j - 3:3 * j + 6] += jjblock/2

    return hbb

def abHessCalcSlow(row, col, bcoords, coords, bvals, nobetas, kGamma):
    from scipy import sparse
    dof = 3*bcoords.shape[0]
    hab = sparse.lil_matrix((dof, dof))
    for k, (i,j) in enumerate(zip(row, col)):
        if abs(i-j) > 1 or np.any(nobetas==j):
            continue
        bj = bvals[j]
        rvec = bcoords[j] - coords[i]
        g = kGamma[k]
        ijblock, jjblock = alphaBetaBond(rvec, bj)
        jiblock = ijblock.T
        iiblock = -ijblock[:,3:6]
        hab[3*i:3*i+3, 3*j-3:3*j+6] += ijblock
        hab[3 * j - 3:3 * j + 6, 3 * i:3 * i + 3] += jiblock
        hab[3*i:3*i+3, 3*i:3*i+3] += iiblock
        hab[3 * j - 3:3 * j + 6, 3 * j - 3:3 * j + 6] += jjblock

    return hab

def addSulfideBonds(sulfs, kirch):
    sCoords = sulfs.getCoords()
    from sklearn.neighbors import BallTree, radius_neighbors_graph
    tree = BallTree(sCoords)
    adjacency = radius_neighbors_graph(tree, 3.0, n_jobs=-1)
    print(adjacency.nnz)
    sNodes = sulfs.getData('nodeid')
    kirch = kirch.tocsr()
    for i, n in enumerate(sNodes):
        neighbors = np.nonzero(adjacency[i,:]==1)
        print(neighbors)
        kirch[i, neighbors] = kirch[i, neighbors]*100

    return kirch.tocoo()


def addIonBonds(atoms, kirch, dists):
    anion = atoms.select('resname ASP GLU')
    cation = atoms.select('resname ASP GLU')

    for i, at in enumerate(atoms.iterAtoms()):
        if at.getData('resname') == 'ASP or GLU':
            neighbors = dists[i, :] <= 4
            print(np.count_nonzero(neighbors))
            kirch[i, neighbors] = kirch[i, neighbors] * 100

    return kirch

def buildChemENM(capsid):
    from scipy import sparse
    import numpy as np
    from sklearn.neighbors import BallTree, radius_neighbors_graph, kneighbors_graph
    import biotite.structure as struc
    calphas = capsid[capsid.atom_name == 'CA']
    print("Number of protein residues:", struc.get_residue_count(capsid))
    coords = calphas.coord #struc.apply_residue_wise(capsid, capsid.coord, np.average, axis=0)
    n_atoms = coords.shape[0]

    from bondStrengths import detect_disulfide_bonds
    sulfs = detect_disulfide_bonds(capsid)
    print(sulfs.shape[0])

    from bondStrengths import detect_salt_bridges
    salts = detect_salt_bridges(capsid)
    print(salts.shape)

    from bondStrengths import hbondSites
    hbonds = hbondSites(capsid)
    print(hbonds.shape[0])

    print('# Atoms ',n_atoms)
    dof = n_atoms * 3

    tree = BallTree(coords)
    kirch = radius_neighbors_graph(tree, 7.5, mode='distance', n_jobs=-1)
    kc = kirch.tocoo().copy()
    kc.sum_duplicates()
    kirch = kirchChem(kc.tocsr(), hbonds, sulfs, salts)
    # kirch = betaTestKirch(coords)
    print('nonzero values: ', kirch.nnz)
    dg = np.array(kirch.sum(axis=0))
    kirch.setdiag(-dg[0])
    kirch.sum_duplicates()
    print(kirch.data)

    if model=='anm':
        kc = kirch.tocoo().copy()
        kc.sum_duplicates()
        hData = hessCalc(kc.row, kc.col, kirch.data, coords)
        indpt = kirch.indptr
        inds = kirch.indices
        hessian = sparse.bsr_matrix((hData, inds, indpt), shape=(dof,dof)).tocsr()
        hessian = fanm*hessian + (1-fanm)*sparse.kron(kirch, np.identity(3))
    else:
        hessian = kirch.copy()
    print('done constructing matrix')
    return kirch, hessian

def kirchChem(kirch, hbonds, sulfbonds, salts):
    kg = kirch.copy()
    kg.data = -1/kg.data**2

    for ij in hbonds:
        i, j = (ij[0], ij[1])
        kg[i,j] = -10

    for ij in salts:
        i, j = (ij[0], ij[1])
        kg[i,j] = -10

    for ij in sulfbonds:
        i, j = (ij[0], ij[1])
        kg[i,j] = -100

    kg.data = np.where(kg.data<=-1/(3**2), -100, kg.data)
    return kg