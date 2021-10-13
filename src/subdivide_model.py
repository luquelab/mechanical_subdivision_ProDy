import os
import time

from prody import *
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn
import psutil




def subdivide_model(pdb, cluster_range, model_in = None ,calphas_in=None, type = 'anm'):
    print(os.getcwd())
    os.chdir("../../results/models")
    print('Loading Model')
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
    print('Calculating Distance Fluctuations')
    start = time.time()
    distFlucts = calcDistFlucts(model, norm=False)
    print('Memory Usage: ', psutil.virtual_memory().percent)
    end = time.time()
    print(end - start, ' Seconds')
    print('Calculating Similarity Matrix')
    n = distFlucts.shape[0]
    start = time.time()
    nearestNeighs = np.full((n, n), True, dtype=bool)
    np.fill_diagonal(nearestNeighs, False)
    dist = buildDistMatrix(calphas.getCoords())
    nearestNeighs &= (dist <= 10.0)
    nnDistFlucts = distFlucts[nearestNeighs]
    sigma = 1 / (2 * np.mean(nnDistFlucts) ** 2)
    sims = np.exp(-sigma * distFlucts * distFlucts)
    print('Memory Usage: ', psutil.virtual_memory().percent)
    end = time.time()
    print(end - start, ' Seconds')

    def embedding(n_evecs, sims):
        print('Performing Spectral Embedding')
        from sklearnex import patch_sklearn
        patch_sklearn()
        from sklearn.manifold import spectral_embedding
        X_transformed = spectral_embedding(sims, n_components=n_evecs, drop_first=False, eigen_solver='amg')
        print('Memory Usage: ', psutil.virtual_memory().percent)
        return X_transformed

    def kmed_embedding(n_range, maps):
        print('Clustering Embedded Points')

        from sklearnex import unpatch_sklearn
        unpatch_sklearn()
        from sklearn_extra.cluster import KMedoids
        from sklearn.cluster import KMeans

        from sklearn.metrics import silhouette_score

        labels = []
        scores_km = []
        for n in range(len(n_range)):
            n_clusters = n_range[n]
            print('Clusters: ' + str(n_clusters))

            kmed = KMeans(n_clusters=n_clusters).fit(maps[:, :n_clusters])
            labels.append(kmed.labels_)

            print('Scoring')
            testScore = silhouette_score(maps[:, :n_clusters], kmed.labels_)
            scores_km.append(testScore)
            print('Memory Usage: ', psutil.virtual_memory().percent)

            print('Saving Results')
            domains = kmed.labels_
            nc = str(n_range[n])
            writePDB(pdb + '_' + nc + '_domains.pdb', calphas, beta=domains)

        return labels, scores_km

    print(os.getcwd())
    os.chdir("../../results/subdivisions")
    print('Spectral Clustering')
    n_range = cluster_range
    n_evecs = max(n_range)
    start = time.time()
    maps = embedding(n_evecs, sims)
    end = time.time()
    print(end - start, ' Seconds')
    start = time.time()
    labels, scores = kmed_embedding(n_range, maps)
    end = time.time()
    print(end - start, ' Seconds')



    print('Plotting')
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.scatter(n_range, scores, marker='D', label=pdb)
    ax.plot(n_range, scores)
    ax.axvline(x=n_range[np.argmax(scores)], label='Best Score', color='black')
    nc = str(n_range[np.argmax(scores)])
    ax.set_xticks(n_range)
    ax.set_xlabel('n_clusters')
    ax.set_ylabel('Silhouette Score')
    ax.legend()
    fig.tight_layout()
    print(pdb + '_' + nc + '_domains.png')
    plt.savefig(pdb + '_' + nc + '_domains.png')
    plt.show()

    return calphas, labels
