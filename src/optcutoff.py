import numpy as np
import numba as nb

#@nb.njit()
def fastFlucts(evals, evecs, n_modes):
    n_atoms = evecs.shape[0]
    flucts = np.zeros(n_atoms)
    for i in range(n_modes):
        flucts += 1/evals[i]*evecs[:,i]**2

    return flucts


#@nb.njit(nb.float64(nb.float64[:], nb.float64[:,:]))
def springFit(bfactors, sqFlucts):
    a, _, _, _ = np.linalg.lstsq(sqFlucts, bfactors)
    return a[0]

def fluctFit(evals, evecs, bfactors):
    coeffs = []
    ks = []
    flucts = []
    for n_modes in range(len(evals)):
        if n_modes==0:
            continue
        c, k, fluct = costFunc(evals, evecs, bfactors, n_modes)
        coeffs.append(c)
        ks.append(k)
        flucts.append(fluct)
    nModes = np.argmax(coeffs)+1
    coeff = np.max(coeffs)
    kbest = ks[nModes-1]
    fluct = flucts[nModes-1]
    return nModes, coeff, kbest, fluct





# @nb.njit()
def costFunc(evals, evecs, bfactors, n_modes):
    sqFlucts = fastFlucts(evals,evecs,n_modes)
    if sqFlucts.shape[0] != bfactors.shape[0]:
        print('anm')
        sqFlucts = np.reshape(sqFlucts, (-1, 3)).sum(axis=-1)

    k = springFit(bfactors, sqFlucts[:,np.newaxis])
    scaledFlucts = k*sqFlucts
    c = np.corrcoef(bfactors,scaledFlucts)[1,0]
    return c, k, scaledFlucts
