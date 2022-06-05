def realFlucts(nc, labels):
    import numpy as np
    from make_model import getPDB
    from settings import pdb, model, n_modes
    from make_model import loadModes

    evals, evecs, kirch = loadModes(pdb, n_modes)


    rigidities, fullRigidities, mobility = loopFlucts(evals, evecs.copy(), labels, model)

    #from prody import writePDB
    # calphas = capsid.select('calpha')
    # calphas.setBetas(fullRigidities)
    # writePDB('../results/subdivisions/' + pdb + '/' + pdb + '_' + str(nc) + '_rigidtest.pdb', calphas, beta=rigidities,
    #          hybrid36=True)
    return rigidities, fullRigidities, mobility

def loopFlucts(evals, evecs, labels, model):
    import numpy as np
    from pythRigidity import fastFlucts, clusterFlucts
    n_evals = evals.shape[0]
    sqFlucts = fastFlucts(evecs, model)

    if model == 'anm':
        evecs = evecs.reshape((-1, 3, n_evals))

    n_clusters = np.max(labels) + 1
    natoms = evecs.shape[0]

    fullRigidities = np.zeros(natoms)

    evecs = evecs * 1 / np.sqrt(evals)

    rigidities = np.zeros_like(fullRigidities)
    mobilities = np.zeros_like(fullRigidities)


    for i in range(n_clusters):
        mask = (labels == i)
        if not np.any(mask):
            print('Some clusters unassigned')
            flucts = np.array([0])
            cFlucts = np.array([0])
        else:
            cVecs = evecs[mask].copy()
            cFlucts = sqFlucts[mask].copy()
            flucts = clusterFlucts(cVecs, cFlucts)
        totalFlucts = flucts.sum()
        mobility = np.sum(np.sqrt(cFlucts))
        rigidities[mask] = totalFlucts
        fullRigidities[mask] = flucts
        mobilities[mask] = mobility
    return rigidities, fullRigidities, mobilities