import numpy as np


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
    nsvar = (var + (x - mean)*(x - nmean))
    return nmean, nsvar

def calcSamples(coords, evals, evecs, n_samples, row, col):
    for i in range(n_samples):
        randvecs = sampleVecs(evals, evecs)
        newCoords = coords + randvecs.reshape(-1,3)
        newDists = dists(newCoords, row, col)
        if i == 0:
            mean = newDists
            svar = np.zeros_like(newDists)
        else:
            mean, svar = incrementalCalc(mean, svar, newDists, i)
    var = svar/n_samples
    return var


def dists(coords, row, col):
    data = np.zeros_like(row)
    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        r = coords[j] - coords[i]
        data[k] = np.sqrt(np.dot(r,r))
    return data