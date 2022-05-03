import numpy as np

#pythran export pairDistsP(float, float[:], float[:,:], float[:,:])
def pairDistsP(sqFluct, evals, evi, evj):
    n_e = evals.shape[0]
    cij = 0
    for n in range(n_e):
        l = evals[n]
        cij += 1 / l * np.dot(evi[:,n], evj[:,n])
    d = sqFluct - 2*cij
    return d

#pythran export clusterFluctsP(float[:], float[:,:,:], float[:])
def clusterFluctsP(evals, evecs, sqFlucts):
    c_size = evecs.shape[0]
    flucts = np.zeros(c_size)
    for i in range(c_size):
        evi = evecs[i,:,:]
        for j in range(c_size):
            if i==j:
                continue
            evj = evecs[j,:,:]
            sqFluct = sqFlucts[i] + sqFlucts[j]
            flucts[i] += np.sqrt(pairDistsP(sqFluct, evals, evi, evj))
    return flucts/(2*c_size)

#pythran export fastFlucts(float[:], float[:,:], int)
def fastFlucts(evals, evecs, n_modes):
    n_atoms = evecs.shape[0]
    flucts = np.zeros(n_atoms)
    for i in range(n_modes):
        flucts += 1/evals[i]*evecs[:,i]**2
    return flucts


#pythran export pairDistsP(float[:], float, float)
def pairDistsG(evals, evi, evj):
    n_e = evals.shape[0]
    cij = 0
    sq1 = 0
    sq2 = 0
    for n in range(n_e):
        l = evals[n]
        sq1 += 1/l*evi**2
        sq2 += 1/l*evj**2
        cij += 1 / l * evi*evj
    d = sq1 + sq2 - 2*cij
    return d

#pythran export clusterFluctsP(float[:], float[:,:])
def clusterFluctsG(evals, evecs):
    c_size = evecs.shape[0]
    flucts = np.zeros(c_size)
    for i in range(c_size):
        evi = evecs[i,:]
        for j in range(c_size):
            if i==j:
                continue
            evj = evecs[j,:]
            flucts[i] += np.sqrt(pairDistsG(evals, evi, evj))
    return flucts/(2*c_size)

#pythran export fastFlucts(float[:], float[:,:], int)
def fastFlucts(evals, evecs, n_modes):
    n_atoms = evecs.shape[0]
    flucts = np.zeros(n_atoms)
    for i in range(n_modes):
        flucts += 1/evals[i]*evecs[:,i]**2
    return flucts

def loopFluctsG(evals, evecs, labels):
    n_clusters = np.max(labels)+1
    fullRigidities = np.zeros_like(evecs)
    rigidities = np.zeros_like(evecs)
    mobilities = np.zeros_like(evecs)
    n_evals = evals.shape[0]
    sqFlucts = fastFlucts(evals, evecs, n_evals)
    for i in range(n_clusters):
        mask = (labels == i)
        if not np.any(mask):
            print('Some clusters unassigned')
            rigidities[i] += 0
        else:
            cVecs = evecs[mask, :].copy()
            cFlucts = sqFlucts[mask].copy()
            flucts = clusterFluctsG(evals, cVecs)
            totalFlucts = flucts.mean()
            mobility = np.mean(np.sqrt(cFlucts))
            rigidities[mask] = totalFlucts
            fullRigidities[mask] = flucts
            mobilities[mask] = mobility
    return rigidities, fullRigidities, mobilities