import numba as nb
import numpy as np

def buildENM(atoms, coords, cutoff, model='gnm', gamma=1., **kwargs):
    from scipy import sparse
    import numpy as np
    from sklearn.neighbors import BallTree, radius_neighbors_graph
    from input import cbeta
    from prody import performDSSP

    n_atoms = coords.shape[0]
    dof = n_atoms * 3


    segs = np.array(atoms.getSegnames())
    chains = np.array(atoms.getChids())
    rnums = np.array(atoms.getResnums())

    tree = BallTree(coords)
    kirch = radius_neighbors_graph(tree, cutoff, mode='distance', n_jobs=-1)
    print(kirch.data)
    kc = kirch.tocoo()
    kirch = kirchGamma(kc, atoms, d2=True, flexibilities=False, cbeta=cbeta)
    print(kirch.data)
    dg = np.array(kirch.sum(axis=0))

    kirch.setdiag(-dg[0])
    if model=='anm':
        hData, hDiags = hessCalc(kirch.row, kirch.col, kirch.data, coords, segs, chains, rnums)
        kirch = kirch.tocsr()
        indpt = kirch.indptr
        inds = kirch.indices
        hessian = sparse.bsr_matrix((hData, inds, indpt), shape=(dof,dof))
    else:
        hessian = kirch.copy()

    return kirch, hessian



def kirchGamma(kirch, atoms, **kwargs):
    kg = kirch.copy()

    if 'd2' in kwargs and kwargs['d2']:
        print('d2 mode')
        kg.data = -1/(kg.data**2)
    else:
        kg.data = -np.ones_like(kg.data)

    if 'flexibilities' in kwargs and kwargs['flexibilities']:
        flex = flexes(1/atoms.getBetas())
        fl = cooOperation(kirch.row, kirch.col, kirch.data, flexFunc, flex)
        kg.data = kg.data*fl

    if 'cbeta' in kwargs and kwargs['cbeta']:
        names = atoms.getNames()
        abgamma = np.array([0 if x == 'CA' else np.sqrt(0.5) for x in names])
        abg = cooOperation(kirch.row, kirch.col, kirch.data, abFunc, abgamma)
        kg.data = kg.data * abg

    return kg


@nb.njit()
def cooOperation(row, col, data, func, arg):
    r = np.copy(data)
    for n, (i, j, v) in enumerate(zip(row, col, data)):
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


def sequenceNeighbors(atoms):
    resIds = atoms.getRes

@nb.njit()
def gammaStruct(d2, s1, s2, ch1, ch2, r1, r2):
    if s1==s2 and ch1 == ch2:
        sij = np.abs(r1 - r2)
        if sij <= 3 and sij > 0:
            return 1/(sij**2)
        else:
            return 1/d2
    else:
        return 1/d2


@nb.njit()
def hessCalc(row, col, kGamma, coords, segs, chains, rnums):
    n_atoms = coords.shape[0]
    hData = np.zeros((row.shape[0], 3, 3))
    hDiags = np.zeros((n_atoms, 3, 3))
    dinds = np.nonzero(row==col)[0]
    for k, (i,j) in enumerate(zip(row,col)):
        if i==j:
            continue
        dvec = coords[j] - coords[i]
        d2 = np.dot(dvec, dvec)
        g = kGamma[k] #gammaFunc(d2, flex[i], flex[j], abgamma[i]+abgamma[j])
        #g = gammaStruct(d2, segs[i], segs[j], chains[i], chains[j], rnums[i], rnums[j])
        hblock = np.outer(dvec, dvec) * (g / d2)
        hData[k,:,:] = hblock
        hData[dinds[i],:,:] += -hblock/2
        hData[dinds[j],:,:] += -hblock/2

    return hData, hDiags


