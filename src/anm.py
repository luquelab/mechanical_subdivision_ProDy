

def hessBuild(atoms, coords, cutoff, gamma=1., **kwargs):
    from scipy import sparse
    import numpy as np
    fl = True
    print(atoms.getNames()[0])
    #from itertools import izip

    n_atoms = coords.shape[0]

    dof = n_atoms * 3


    from sklearn.neighbors import BallTree, radius_neighbors_graph
    tree = BallTree(coords)
    #test = tree.query_radius(coords, cutoff)


    kirch = -radius_neighbors_graph(tree, cutoff, n_jobs=-1)
    dg = np.array(kirch.sum(axis=0))

    kirch.setdiag(-dg[0])

    kc = kirch.tocoo()

    # hdata = np.zeros((n_atoms, n_atoms, 3, 3))
    if fl:
        flex = flexes(1/atoms.getBetas())
    else:
        flex = np.ones(n_atoms)
    hData, hDiags = hessCalc(kc.row, kc.col, coords, flex)
    indpt = kirch.indptr
    inds = kirch.indices
    hessian = sparse.bsr_matrix((hData, inds, indpt), shape=(dof,dof))

    #print(tree.get_arrays()[1])
    return kirch, hessian


import numba as nb
import numpy as np
@nb.njit()
def hessCalc(row, col, coords, flex):
    n_atoms = coords.shape[0]
    hData = np.zeros((row.shape[0], 3, 3))
    hDiags = np.zeros((n_atoms, 3, 3))
    dinds = np.nonzero(row==col)[0]
    for k, (i,j) in enumerate(zip(row,col)):
        if i==j:
            #dinds[i] = k
            continue
        dvec = coords[j] - coords[i]
        d2 = np.dot(dvec, dvec)
        g = gammaFunc(d2, flex[i], flex[j])
        hblock = np.outer(dvec, dvec) * (- g / d2)
        hData[k,:,:] = hblock
        hData[dinds[i],:,:] += -hblock/2
        hData[dinds[j],:,:] += -hblock/2

    # hData[dinds,:,:] = hDiags
    return hData, hDiags

def flexes(bfactors):
    return bfactors/np.mean(bfactors)

@nb.njit()
def gammaFunc(d2, f1, f2):
    return 1/d2 * np.sqrt(f1*f2)