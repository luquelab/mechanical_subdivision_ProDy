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
    from sklearn.linear_model import HuberRegressor
    #a, _, _, _ = np.linalg.lstsq(sqFlucts, bfactors)
    #a = springFit2(bfactors, sqFlucts)
    huber = HuberRegressor(fit_intercept=False).fit(sqFlucts, bfactors)
    a = huber.coef_
    return a

def springFit2(bfactors, sqFlucts):
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda k: l1Cost(bfactors, sqFlucts, k))
    return res.x


@nb.njit()
def l1Cost(bfactors, sqFlucts, a):
    return np.sum(np.abs(bfactors - a * sqFlucts))

def fluctFit(evals, evecs, bfactors):
    fitmodes = False
    coeffs = []
    ks = []
    flucts = []
    from settings import n_modes
    minModes = int(0.1*n_modes)
    if fitmodes:
        for n_modes in range(len(evals)):
            if n_modes < minModes:
                continue
            c, k, fluct = costFunc(evals, evecs, bfactors, n_modes)
            coeffs.append(c)
            ks.append(k)
            flucts.append(fluct)
        nModes = np.argmax(coeffs)+1
        coeff = np.max(coeffs)
        kbest = ks[nModes-1]
        fluct = flucts[nModes-1]
        return int(nModes+minModes), coeff, kbest[0], fluct
    else:
        n_modes = evals.shape[0]
        c, k, fluct = costFunc(evals, evecs, bfactors, n_modes)
        return n_modes, c, k[0], fluct


# @nb.njit()
def costFunc(evals, evecs, bfactors, n_modes):
    sqFlucts = fastFlucts(evals,evecs,n_modes)
    if sqFlucts.shape[0] != bfactors.shape[0]:
        sqFlucts = np.reshape(sqFlucts, (-1, 3)).sum(axis=-1)

    k = springFit(bfactors, sqFlucts[:,np.newaxis])
    scaledFlucts = k*sqFlucts
    c = np.corrcoef(bfactors,scaledFlucts)[1,0]
    return c, k, scaledFlucts