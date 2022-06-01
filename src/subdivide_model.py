import os
import time

from prody import *
# from sklearnex import patch_sklearn
# patch_sklearn()
import numpy as np
import matplotlib.pyplot as plt
import psutil
from scipy import sparse
from settings import *


def subdivide_model():
    print('Loading Model')

    sims = sparse.load_npz('../results/models/' + pdb + 'sims.npz')
    calphas = loadAtoms('../results/models/' + 'calphas_' + pdb + '.ag.npz')


    if not os.path.exists('../results/subdivisions/' + pdb):
        os.mkdir('../results/subdivisions/' + pdb)

    print('Spectral Clustering')

    n_range = np.arange(cluster_start, cluster_stop + cluster_step, cluster_step)
    n_evecs = max(n_range)

    if mode == 'clustering':
        print('Plotting')
        from score import plotScores
        from score import clustFlucts
        # clustFlucts(labels[n_best], pdb)
        plotScores(pdb, n_range, save=True)

        return -1

    if mode=='embedding' and os.path.exists('../results/models/' + pdb + 'embedding.npy'):
        maps = np.load('../results/models/' + pdb + 'embedding.npy')
        if maps.shape[1] < cluster_stop:
            print('Insufficient Eigenvectors For ' + str(cluster_stop) + ' Clusters, continuing with ' + str(maps.shape[1]) + 'as cluster_stop')
            n_range = n_range[:maps.shape[1]].copy()
    else:
        if mode=='embedding':
            print('No saved embedding found, rebuilding')
        else:
            print('Starting Spectral Embedding')
        start = time.time()
        maps = embedding(n_evecs, sims)
        end = time.time()
        print('Time Of Spectral Embedding: ', end - start, ' Seconds')
        from sklearn.preprocessing import normalize
        normalize(maps, copy=False)
        np.save('../results/models/' + pdb + 'embedding.npy', maps)

    print('Starting Eigenvector Clustering')
    start = time.time()
    labels, scores, var, ntypes, inerts = cluster_embedding(n_range, maps, calphas, cluster_method)
    end = time.time()
    print('Time Of Clustering: ', end - start, ' Seconds')

    n_best = np.argmax(scores)

    print('Plotting')
    from score import plotScores
    from score import clustFlucts
    clustFlucts(labels[n_best], pdb)
    plotScores(pdb, n_range, save=True)

    #return -1

def embedding(n_evecs, sims):
    from spectralStuff import spectral_embedding
    from make_model import evPlot
    print('Performing Spectral Embedding')
    from scipy.sparse.csgraph import connected_components
    print(connected_components(sims))
    X_transformed, evals = spectral_embedding(sims, n_components=n_evecs, drop_first=False, eigen_solver = 'lobpcg', norm_laplacian=False)
    print('Memory Usage: ', psutil.virtual_memory().percent)
    evPlot(np.ediff1d(evals), X_transformed)
    evPlot(evals, X_transformed)
    return X_transformed

def cluster_embedding(n_range, maps, calphas, method):
    print('Clustering Embedded Points')

    from sklearn.cluster import k_means
    #from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
    #from sklearn.metrics import pairwise_distances
    # from sklearn_extra.cluster import KMedoids

    from sklearn.metrics import silhouette_score
    from sklearn.metrics import davies_bouldin_score
    from score import median_score, cluster_types
    from score import calcCentroids
    from sklearn.preprocessing import normalize



    labels = []
    scores = []
    variances = []
    numtypes = []
    inerts = []
    print('mapshape', maps.shape)
    for n in range(len(n_range)):
        n_clusters = n_range[n]
        emb = maps[:, :n_clusters].copy()
        normalize(emb, copy=False)

        print('Clusters: ' + str(n_clusters))
        print(method)
        print('emshape', emb.shape)
        if method == 'discretize':
            # loop = True
            # while loop:
            label = discretize(emb)
            centroids, loop = calcCentroids(emb, label, n_clusters)
            inert = 0
        elif method == 'kmeans':
            centroids, label, inert, n_iter = k_means(emb, n_clusters=n_clusters, n_init=100, tol=0,
                                                  return_n_iter=True)
        elif method == 'both':
            label = discretize(emb)
            centroids = calcCentroids(emb, label, n_clusters)
            normalize(centroids, copy=False)
            centroids, label, inert, n_iter = k_means(emb, n_clusters=n_clusters, init=discreteInit,
                                                      return_n_iter=True)

        else:
            print('method should be kmeans or discretize. Defaulting to kmeans')

        normalize(centroids, copy=False)
        # normalize(centroids)
        labels.append(label)
        cl = np.unique(label)
        print(cl.shape)
        end1 = time.time()

        print('Scoring')
        from rigidity import realFlucts
        testScore = median_score(emb, centroids)
        #rigidities, _, _ = realFlucts(n_clusters, label)
        #inert = np.sum(rigidities)

        scores.append(testScore)
        var, ntypes = cluster_types(label)
        variances.append(var)
        numtypes.append(ntypes)
        inerts.append(inert)
        print('Memory Usage: ', psutil.virtual_memory().percent)

        print('Saving Results')
        nc = str(n_range[n])
        np.savez('../results/subdivisions/' + pdb + '/' + pdb + '_' + nc + '_results', labels=label, score=testScore,
                 var=var, ntypes=ntypes, n=n, method=cluster_method, inertia=inert)

    best = np.argpartition(scores, -5)[-5:]  # indices of 4 best scores
    for ind in best:
        saveSubdivisions(labels[ind], n_range[ind])

    return labels, scores, variances, numtypes, inerts

def saveSubdivisions(labels, nsub):
    from settings import pdb
    from make_model import getPDB
    from prody import writePDB, saveAtoms
    capsid, _, _ = getPDB(pdb)
    capsid.setData('clust', 1)
    nodes = capsid.getData('nodeid')
    nmax = np.max(nodes)
    for at in capsid.iterAtoms():
        node = at.getData('nodeid')
        at.setData('clust', labels[node % nmax])
        at.setBeta(labels[node % nmax])
    writePDB('../results/subdivisions/' + pdb + '/' + pdb + '_' + str(nsub) + '_domains.pdb', capsid,hybrid36=True)
    saveAtoms(capsid, '../results/subdivisions/' + pdb + '/' + pdb + '_' + str(nsub) + '_domains_atoms')
    return capsid



def discreteInit(vectors, n_clusters, *, copy=False, max_svd_restarts=30, n_iter_max=20, random_state=None):
    from score import calcCentroids
    label = discretize(vectors)
    centroids = calcCentroids(vectors, label, n_clusters)
    return centroids

def discretize(
    vectors, n_clusters=10, *, copy=False, max_svd_restarts=300, n_iter_max=2000, random_state=None
):
    """Search for a partition matrix which is closest to the eigenvector embedding.
    This implementation was proposed in [1]_.
    Parameters
    ----------
    vectors : array-like of shape (n_samples, n_clusters)
        The embedding space of the samples.
    copy : bool, default=True
        Whether to copy vectors, or perform in-place normalization.
    max_svd_restarts : int, default=30
        Maximum number of attempts to restart SVD if convergence fails
    n_iter_max : int, default=30
        Maximum number of iterations to attempt in rotation and partition
        matrix search if machine precision convergence is not reached
    random_state : int, RandomState instance, default=None
        Determines random number generation for rotation matrix initialization.
        Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.
    References
    ----------
    .. [1] `Multiclass spectral clustering, 2003
           Stella X. Yu, Jianbo Shi
           <https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf>`_
    Notes
    -----
    The eigenvector embedding is used to iteratively search for the
    closest discrete partition.  First, the eigenvector embedding is
    normalized to the space of partition matrices. An optimal discrete
    partition matrix closest to this normalized embedding multiplied by
    an initial rotation is calculated.  Fixing this discrete partition
    matrix, an optimal rotation matrix is calculated.  These two
    calculations are performed until convergence.  The discrete partition
    matrix is returned as the clustering solution.  Used in spectral
    clustering, this method tends to be faster and more robust to random
    initialization than k-means.
    """

    from scipy.sparse import csc_matrix
    from scipy.linalg import LinAlgError
    from sklearn.utils import check_random_state, as_float_array

    random_state = check_random_state(random_state)

    vectors = as_float_array(vectors, copy=copy)

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    # Normalize the eigenvectors to an equal length of a vector of ones.
    # Reorient the eigenvectors to point in the negative direction with respect
    # to the first element.  This may have to do with constraining the
    # eigenvectors to lie in a specific quadrant to make the discretization
    # search easier.
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    # Normalize the rows of the eigenvectors.  Samples should lie on the unit
    # hypersphere centered at the origin.  This transforms the samples in the
    # embedding space to the space of partition matrices.
    vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]

    svd_restarts = 0
    has_converged = False

    # If there is an exception we try to randomize and rerun SVD again
    # do this max_svd_restarts times.
    while (svd_restarts < max_svd_restarts) and not has_converged:

        # Initialize first column of rotation matrix with a row of the
        # eigenvectors
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        # To initialize the rest of the rotation matrix, find the rows
        # of the eigenvectors that are as orthogonal to each other as
        # possible
        c = np.zeros(n_samples)
        for j in range(1, n_components):
            # Accumulate c to ensure row is as orthogonal as possible to
            # previous picks as well as current one
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components),
            )

            t_svd = vectors_discrete.T * vectors

            try:
                U, S, Vh = np.linalg.svd(t_svd)
            except LinAlgError:
                svd_restarts += 1
                print("SVD did not converge, randomizing and trying again")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if (abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max):
                has_converged = True
            else:
                # otherwise calculate rotation and continue
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError("SVD did not converge")
    return labels