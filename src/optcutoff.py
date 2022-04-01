import numpy as np

def fastFlucts(evals, evecs, n_modes):
    n_atoms = evecs.shape[0]
    flucts = np.zeros(n_atoms)
    for i in range(n_modes):
        flucts += 1/evals[i]*evecs[:,i]**2

    return flucts

def springFit(bfactors, sqFlucts):
    a = np.linalg.lstsq(sqFlucts[:, np.newaxis], bfactors, rcond=None)[0]
    return a, a*sqFlucts

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






def costFunc(evals, evecs, bfactors, n_modes):
    sqFlucts = fastFlucts(evals,evecs,n_modes)
    k, scaledFlucts = springFit(bfactors, sqFlucts)
    c = np.corrcoef(bfactors,scaledFlucts)[1,0]
    return c, k, scaledFlucts
