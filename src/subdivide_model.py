import os
import time

from prody import *
from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import matplotlib.pyplot as plt
import psutil
from scipy import sparse


def subdivide_model(pdb, cluster_start, cluster_stop, cluster_step):

    print('Loading Model')
    sims = sparse.load_npz('../results/models/' + pdb + 'sims.npz')
    calphas = loadAtoms('../results/models/' + 'calphas_' + pdb + '.ag.npz')



    def embedding(n_evecs, sims):
        print('Performing Spectral Embedding')
        from sklearn.manifold import spectral_embedding
        X_transformed = spectral_embedding(sims, n_components=n_evecs, drop_first=False)
        print('Memory Usage: ', psutil.virtual_memory().percent)
        return X_transformed

    def kmean_embedding(n_range, maps):
        print('Clustering Embedded Points')

        from sklearn.cluster import k_means
        from sklearn.metrics import pairwise_distances
        from sklearn_extra.cluster import KMedoids

        from sklearn.metrics import silhouette_score
        from sklearn.metrics import davies_bouldin_score


        labels = []
        scores_km = []
        for n in range(len(n_range)):
            n_clusters = n_range[n]
            print('Clusters: ' + str(n_clusters))

            # kmed = KMeans(n_clusters=n_clusters, n_init=200, tol=1e-8).fit(maps[:, :n_clusters])
            # labels.append(kmed.labels_)
            _, label, _ = k_means(maps[:, :n_clusters], n_clusters=n_clusters, n_init=50)
            # kmed = KMedoids(n_clusters=n_clusters).fit(maps[:, :n_clusters])
            # _, label, _ = spherical_k_means(maps[:, :n_clusters], n_clusters=n_clusters)

            print('Scoring')
            testScore = silhouette_score(maps[:, :n_clusters], label)
            scores_km.append(testScore)
            print('Memory Usage: ', psutil.virtual_memory().percent)

            print('Saving Results')
            nc = str(n_range[n])
            writePDB('../results/subdivisions/' + pdb + '/' + pdb + '_' + nc + '_domains.pdb', calphas, beta=label, hybrid36=True)
            np.savez('../results/subdivisions/' + pdb + '/' + pdb + '_' + nc + '_results', labels=label, score=testScore)

        return labels, scores_km

    def clara(n_r, maps):
        print('Clustering Embedded Points')
        from pyclustering.cluster.clarans import clarans

        from sklearn.metrics import silhouette_score


        labels = []
        scores_km = []
        for n in range(len(n_r)):
            n_clusters = n_r[n]
            M = maps[:, :n_clusters]
            print('Clusters: ' + str(n_clusters))

            cl = clarans(M.tolist(), n_clusters, 3, 5)
            cl.process()
            label = cl.get_clusters()
            print(label)

            print('Scoring')
            testScore = silhouette_score(maps[:, :n_clusters], label, metric='cosine')
            scores_km.append(testScore)
            print('Memory Usage: ', psutil.virtual_memory().percent)

            print('Saving Results')
            nc = str(n_range[n])
            writePDB('../results/subdivisions/' + pdb + '/' + pdb + '_' + nc + '_domains.pdb', calphas, beta=label, hybrid36=True)
            np.savez('../results/subdivisions/' + pdb + '/' + pdb + '_' + nc + '_results', labels=label, score=testScore)

        return labels, scores_km

    from sklearn.preprocessing import normalize

    if not os.path.exists('../results/subdivisions/' + pdb):
        os.mkdir('../results/subdivisions/' + pdb)

    print('Spectral Clustering')
    n_range = np.arange(cluster_start, cluster_stop, cluster_step)
    n_evecs = max(n_range)

    start = time.time()
    maps = embedding(n_evecs, sims)
    end = time.time()
    print(end - start, ' Seconds')

    normalize(maps, copy=False)

    start = time.time()
    labels, scores = kmean_embedding(n_range, maps)
    end = time.time()
    print(end - start, ' Seconds')

    print('Plotting')
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.scatter(n_range, scores, marker='D', label=pdb)
    ax.plot(n_range, scores)
    ax.axvline(x=n_range[np.argmax(scores)], label='Best Score', color='black')
    nc = str(n_range[np.argmax(scores)])
    ax.set_xticks(np.arange(cluster_start,cluster_stop,4))
    ax.set_xlabel('n_clusters')
    ax.set_ylabel('Silhouette Score')
    ax.legend()
    fig.tight_layout()
    print(pdb + '_' + nc + '_domains.png')
    plt.savefig('../results/subdivisions/' + pdb + '_' + nc + '_domains.png')
    plt.show()

    return calphas, labels
