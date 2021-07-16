import os
from prody import *
from sklearn import cluster
import numpy as np




def subdivide_model(pdb, n_clusters, model_in = None ,calphas_in=None, type = 'anm'):
    print(os.getcwd())
    if model_in is None:
        if type == 'gnm':
            model = loadModel(pdb + '_full.gnm.npz')
        elif type == 'anm':
            model = loadModel(pdb + '_full.anm.npz')
    else:
        model = model_in
    if calphas_in is None:
        calphas = loadAtoms('calphas_' + pdb + '.ag.npz')
    else:
        calphas = calphas_in

    if type == 'anm':
        distFlucts = calcDistFlucts(model, norm=False)
        n = distFlucts.shape[0]
        nearestNeighs = np.full((n, n), True, dtype=bool)
        np.fill_diagonal(nearestNeighs, False)
        dist = buildDistMatrix(calphas.getCoords())
        nearestNeighs &= (dist <= 10.0)
        nnDistFlucts = distFlucts[nearestNeighs]
        sigma = 1 / (2 * np.mean(nnDistFlucts) ** 2)
        sims = np.exp(-sigma * distFlucts * distFlucts)
    elif type == 'gnm':
        sims = model.getKirchhoff()


    test = cluster.SpectralClustering(assign_labels='kmeans',n_clusters=n_clusters,n_init=1000,affinity='precomputed',n_jobs=8).fit(sims)

    domains = test.labels_
    print(len(np.unique(domains)))
    calphas.setData('domain',domains)
    os.chdir("../subdivisions")
    writePDB(pdb + '_domains.pdb',calphas,beta=domains)

    return calphas, domains
