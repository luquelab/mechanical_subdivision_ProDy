import numba as nb
import numpy as np

@nb.njit(parallel=True)
def pairDists(sqFlucts, evals, evecs, i, j):
    n_e = evals.shape[0]
    cij = 0
    for n in range(n_e):
        l = evals[n]
        cij += 1 / l * np.sum(evecs[i,:,n] * evecs[j,:,n])
    d = sqFlucts[i] + sqFlucts[j] - 2*cij
    return d

@nb.njit()
def clusterFlucts(evals, evecs, sqFlucts):
    c_size = evecs.shape[0]
    flucts = np.zeros(c_size)
    for i in range(c_size):
        for j in range(c_size):
            if i==j:
                continue
            flucts[i] += pairDists(sqFlucts, evals, evecs, i, j)
    return flucts, flucts.mean()

@nb.njit()
def loopFlucts(evals, evecs, labels, sqFlucts):
    n_clusters = np.max(labels)
    fullRigidities = np.zeros_like(sqFlucts)
    rigidities = np.zeros_like(sqFlucts)
    n_evals = evals.shape[0]
    evecs = evecs.reshape(-1,3,n_evals)
    print(evecs.shape)
    for i in range(n_clusters):
        mask = (labels == i)
        if not np.any(mask):
            print('Some clusters unassigned')
            rigidities[i] = 0
        else:
            cVecs = evecs[mask, :]
            # cVecs = cVecs.reshape(-1, n_evals)
            cFlucts = sqFlucts[mask]
            flucts, totalFlucts = clusterFlucts(evals, cVecs, cFlucts)
            print(totalFlucts)
            fullRigidities[mask] = flucts
            rigidities[mask] = totalFlucts

    return rigidities, fullRigidities

def realFlucts(nc, pdb):
    #from input import pdb
    data = np.load('../results/subdivisions/' + pdb + '_sqFlucts.npz')
    sqFlucts = data['sqFlucts']
    modes = np.load('../results/models/' + pdb + 'anm' + 'modes.npz')
    evals = modes['evals']
    evecs = modes['evecs']
    results = np.load('../results/subdivisions/' + pdb + '/' + pdb + '_' + str(nc) + '_results.npz')
    labels = results['labels']

    rigidities, fullRigidities = loopFlucts(evals,evecs,labels,sqFlucts)


    from prody import loadAtoms, writePDB
    calphas = loadAtoms('../results/models/' + 'calphas_' + pdb + '.ag.npz')
    writePDB('../results/subdivisions/' + pdb + '/' + pdb + '_' + str(nc) + '_rigidtest.pdb', calphas,
             beta=rigidities,
             hybrid36=True)
    return rigidities, fullRigidities
