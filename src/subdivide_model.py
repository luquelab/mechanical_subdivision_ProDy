import os
import time

from prody import *
from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import matplotlib.pyplot as plt
import psutil
from scipy import sparse
from input import *


def subdivide_model(pdb, cluster_start, cluster_stop, cluster_step):

    print('Loading Model')
    sims = sparse.load_npz('../results/models/' + pdb + 'sims.npz')
    calphas = loadAtoms('../results/models/' + 'calphas_' + pdb + '.ag.npz')



    def embedding(n_evecs, sims):
        print('Performing Spectral Embedding')
        from sklearn.manifold import spectral_embedding
        from scipy.sparse.csgraph import connected_components
        print(connected_components(sims))
        X_transformed = spectral_embedding(sims, n_components=n_evecs, drop_first=False)
        print('Memory Usage: ', psutil.virtual_memory().percent)
        return X_transformed

    def kmean_embedding(n_range, maps):
        print('Clustering Embedded Points')

        from sklearn.cluster import k_means
        from sklearn.metrics import pairwise_distances
        # from sklearn_extra.cluster import KMedoids

        from sklearn.metrics import silhouette_score
        from sklearn.metrics import davies_bouldin_score
        from score import median_score, cluster_types


        labels = []
        scores_km = []
        variances = []
        numtypes = []
        for n in range(len(n_range)):
            n_clusters = n_range[n]
            print('Clusters: ' + str(n_clusters))

            # kmed = KMeans(n_clusters=n_clusters, n_init=200, tol=1e-8).fit(maps[:, :n_clusters])
            # labels.append(kmed.labels_)
            centroids, label, _ = k_means(maps[:, :n_clusters], n_clusters=n_clusters, n_init=10)
            # kmed = KMedoids(n_clusters=n_clusters).fit(maps[:, :n_clusters])
            # _, label, _ = spherical_k_means(maps[:, :n_clusters], n_clusters=n_clusters)

            print('Scoring')
            testScore = median_score(maps[:, :n_clusters], centroids)
            scores_km.append(testScore)
            var, ntypes = cluster_types(label)
            variances.append(var)
            numtypes.append(ntypes)
            print('Memory Usage: ', psutil.virtual_memory().percent)

            print('Saving Results')
            nc = str(n_range[n])
            np.savez('../results/subdivisions/' + pdb + '/' + pdb + '_' + nc + '_results', labels=label, score=testScore, var=var, ntypes=ntypes,n=n)
            labels.append(label)

        best = np.argpartition(scores_km, -5)[-5:]  # indices of 4 best scores
        for ind in best:
            writePDB('../results/subdivisions/' + pdb + '/' + pdb + '_' + str(n_range[ind]) + '_domains.pdb', calphas,
                     beta=labels[ind],
                     hybrid36=True)

        return labels, scores_km, variances, numtypes




    if not os.path.exists('../results/subdivisions/' + pdb):
        os.mkdir('../results/subdivisions/' + pdb)

    print('Spectral Clustering')
    n_range = np.arange(cluster_start, cluster_stop+cluster_step, cluster_step)
    n_evecs = max(n_range)

    start = time.time()
    maps = embedding(n_evecs, sims)
    end = time.time()
    print(end - start, ' Seconds')
    print(maps.shape)
    from sklearn.preprocessing import normalize
    normalize(maps, copy=False)

    start = time.time()
    labels, scores, var, ntypes = kmean_embedding(n_range, maps)
    end = time.time()
    print(end - start, ' Seconds')

    print('Plotting')
    fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    ax[0].scatter(n_range, scores, marker='D', label=pdb)
    ax[0].plot(n_range, scores)
    ax[1].plot(n_range, ntypes)
    ax[1].scatter(n_range, ntypes)
    ax[1].set_ylabel('# of unique clusters')
    ax[2].plot(n_range, var)
    ax[2].scatter(n_range, var)
    ax[2].set_ylabel('Variance In Cluster Size')
    ax[0].axvline(x=n_range[np.argmax(scores)], label='Best Score', color='black')
    nc = str(n_range[np.argmax(scores)])
    ax[0].set_xticks(np.arange(cluster_start, cluster_stop, 1))
    ax[2].set_xlabel('n_clusters')
    ax[0].set_ylabel('Silhouette Score')
    ax[0].legend()
    fig.tight_layout()
    print(pdb + '_' + nc + '_domains.png')
    # plt.savefig('../results/subdivisions/' + pdb + '_' + nc + '_domains.png')
    plt.show()

    return calphas, labels


# unused

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
        np.savez('../results/subdivisions/' + pdb + '/' + pdb + '_' + nc + '_results', labels=label, score=testScore)

    best = np.argpartition(scores_km, -5)[-5:] # indices of 4 best scores
    for ind in best:
        writePDB('../results/subdivisions/' + pdb + '/' + pdb + '_' + n_r[ind] + '_domains.pdb', calphas, beta=labels[ind],
                 hybrid36=True)


    return labels, scores_km