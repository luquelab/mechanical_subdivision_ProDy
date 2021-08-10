import os
from prody import *
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn




def subdivide_model(pdb, n_cluster_min, n_cluster_max, model_in = None ,calphas_in=None, type = 'anm'):
    print(os.getcwd())
    # os.chdir("../results/models")
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

    distFlucts = calcDistFlucts(model, norm=False)
    n = distFlucts.shape[0]
    nearestNeighs = np.full((n, n), True, dtype=bool)
    np.fill_diagonal(nearestNeighs, False)
    dist = buildDistMatrix(calphas.getCoords())
    nearestNeighs &= (dist <= 10.0)
    nnDistFlucts = distFlucts[nearestNeighs]
    sigma = 1 / (2 * np.mean(nnDistFlucts) ** 2)
    sims = np.exp(-sigma * distFlucts * distFlucts)


    from sklearnex import patch_sklearn
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_samples, silhouette_score
    patch_sklearn()

    def spectral_raw(sims, calphas, n_range):
        quality = []
        labels = []
        cluster_types = []
        for n_clusters in n_range:
            print(n_clusters)
            clust = SpectralClustering(n_clusters=n_clusters, n_init=500, affinity='precomputed').fit(sims)
            label = clust.labels_
            labels.append(label)
            __, counts = np.unique(label, return_counts=True)
            cluster_types.append(counts)
        return quality, labels, cluster_types

    n_range = np.arange(n_cluster_min, n_cluster_max)

    quality, labels, cluster_types = spectral_raw(sims, calphas, n_range)

    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    scores = []
    devs = []
    os.chdir("../subdivisions/")
    for n in range(len(n_range)):
        domains = labels[n]
        __, counts = np.unique(domains, return_counts=True)
        devs.append(np.std(counts))
        testScore = silhouette_score(distFlucts, domains, metric='precomputed')
        scores.append(testScore)
        print(len(np.unique(domains)))
        calphas.setData('b', domains)
        nc = str(n_range[n])
        writePDB(pdb + '_' + nc + '_domains.pdb', calphas, beta=domains)

    # cluster_similarity = [np.var(t)/(np.mean(t)) for t in cluster_types]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.scatter(n_range, scores, marker='D', label=pdb)
    ax.plot(n_range, scores)
    ax.axvline(x=n_range[np.argmax(scores)], label='Best Score', color='black')
    ax.set_xticks(n_range)
    ax.set_xlabel('n_clusters')
    ax.set_ylabel('Silhouette Score')
    ax.legend()
    # ax[1].plot([32,60,90], cluster_types)
    # ax[1].set_ylabel('Unique Clusters')
    # fig.tight_layout()
    plt.show()

    return calphas, domains
