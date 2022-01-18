def sampleVecs(evals, evecs):
    nmodes = evals.shape[0]
    import numpy as np
    stds = 1/np.sqrt(evals)
    rands = np.random.randn(nmodes)
    lengths = stds*rands
    vecs = evecs*lengths[:,None]
    return vecs

def incrementalCalc(mean, var, x, n):
    nmean = mean + 1/n *(x - mean)
    nvar = (var + (x - mean)*(x - nmean))/n
    return nmean, nvar

def calcSamples(coords, evals, evecs, n_samples):
    for i in range(n_samples):
