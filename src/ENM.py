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
    kirch = kirchGamma(kc, bfactors, d2=d2, flexibilities=flexibilities, cbeta=cbeta, struct=False).tocsr()
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
        kg.data = -1/(kg.data**2)
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

def betaCarbonModel(kirch, coords):
    from settings import cutoff
    from sklearn.neighbors import BallTree, radius_neighbors_graph
    betaCoords = buildBetas(coords)

    btree = BallTree(betaCoords)
    betaKirch = radius_neighbors_graph(btree, cutoff, mode='distance',  n_jobs=-1)
    abKirch = btree.query_radius(coords, r=cutoff, return_distance=True)


    return abKirch, betaKirch, kirch


@nb.njit()
def buildBetas(coords):
    bcoords = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        if i==0 or i==coords.shape[0]:
            continue
        else:
            r = 2*coords[i,:] - coords[i-1,:]-coords[i+1,:]
            bcoords[i,:] = coords[i,:] + 3.0*(r)/np.linalg.norm(r)
    return bcoords

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



