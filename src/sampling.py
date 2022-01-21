import numpy as np
#pythran export sampleVecs(float64[:], float64[:,:])
def sampleVecs(evals, evecs):
    nmodes = evals.shape[0]
    stds = 1/np.sqrt(evals)
    rands = np.random.randn(nmodes)
    lengths = stds*rands
    vecs = evecs*lengths
    r = np.sum(vecs, axis=1)
    return r
#pythran export incrementalCalc(float64[:], float64[:], float64[:], int32)
def incrementalCalc(mean, var, x, n):
    nmean = mean + 1/n *(x - mean)
    nsvar = (var + (x - mean)*(x - nmean))
    return nmean, nsvar


#pythran export dd(float64[:,:],int32[:],int32[:])
def dd(coords, row, col):
    data = np.zeros_like(row, dtype=float)
    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        r = coords[j] - coords[i]
        data[k] = np.sqrt(np.dot(r,r))
    return data

#pythran export calcSample(float64[:,:], float64[:], float64[:,:], int, int[:], int[:])
def calcSample(coords, evals, evecs, n_samples, row, col):
    mean = dd(coords, row, col)
    svar = np.zeros_like(mean, dtype=float)
    for i in range(n_samples):
        randvecs = sampleVecs(evals, evecs)
        newCoords = coords + randvecs.reshape(-1,3)
        newDists = dd(newCoords, row, col)
        if i == 0:
            mean = newDists
            svar = np.zeros_like(mean, dtype=float)
        else:
            mean, svar = incrementalCalc(mean, svar, newDists, i)
    var = svar/n_samples
    return var