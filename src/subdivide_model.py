import os
import time

from prody import *
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn
import psutil
import numba as nb


def subdivide_model(pdb, cluster_start, cluster_stop, cluster_step, model_in=None, calphas_in=None, type='anm'):
    print(os.getcwd())
    print('Loading Model')
    if model_in is None:
        os.chdir("../../results/models")
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

    @nb.njit(parallel=False)
    def cov(evals, evecs, i, j):
        n_e = evals.shape[0]
        n_d = evecs.shape[1]
        tr1 = 0
        tr2 = 0
        tr3 = 0
        for n in nb.prange(n_e):
            l = evals[n]
            tr1 += 1 / l * (evecs[3 * i, n] * evecs[3 * j, n] + evecs[3 * i + 1, n] * evecs[3 * j + 1, n] + evecs[
                3 * i + 2, n] * evecs[3 * j + 2, n])
            tr2 += 1 / l * (evecs[3 * i, n] * evecs[3 * i, n] + evecs[3 * i + 1, n] * evecs[3 * i + 1, n] + evecs[
                3 * i + 2, n] * evecs[3 * i + 2, n])
            tr3 += 1 / l * (evecs[3 * j, n] * evecs[3 * j, n] + evecs[3 * j + 1, n] * evecs[3 * j + 1, n] + evecs[
                3 * j + 2, n] * evecs[3 * j + 2, n])
        cov = tr1 / np.sqrt(tr2 * tr3)
        return cov

    def con_c(evals, evecs, c, row, col):
        n_d = int(evecs.shape[0] / 3)
        n_e = evals.shape[0]

        for k in range(row.shape[0]):
            i, j = (row[k], col[k])
            c[i, j] = cov(evals, evecs, i, j)
        return c

    def con_d(c, d, row, col):
        for k in range(row.shape[0]):
            i, j = (row[k], col[k])
            d[i, j] = 2 - 2 * c[i, j]
        return d

    from scipy import sparse

    evals = model.getEigvals()
    evecs = model.getEigvecs()
    n_d = int(evecs.shape[0] / 3)

    kirch = model.getKirchhoff().tocoo()

    covariance = sparse.lil_matrix((n_d, n_d))
    df = sparse.lil_matrix((n_d, n_d))
    covariance = con_c(evals, evecs, covariance, kirch.row, kirch.col)
    covariance = covariance.tocsr()
    d = con_d(covariance, df, kirch.row, kirch.col)
    d = d.tocsr()

    nnDistFlucts = np.mean(d.data)

    sigma = 1 / (2 * nnDistFlucts ** 2)
    sims = -sigma * d ** 2
    data = sims.data
    data = np.exp(data)
    sims.data = data

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


        from sklearn.cluster import KMeans
        from sklearn.cluster import k_means

        from sklearn.metrics import silhouette_score

        labels = []
        scores_km = []
        for n in range(len(n_range)):
            n_clusters = n_range[n]
            print('Clusters: ' + str(n_clusters))

            # kmed = KMeans(n_clusters=n_clusters, n_init=200, tol=1e-8).fit(maps[:, :n_clusters])
            # labels.append(kmed.labels_)
            _, label, _ = k_means(maps[:, :n_clusters], n_clusters=n_clusters, n_init=200, tol=1e-8)


            print('Scoring')
            testScore = silhouette_score(maps[:, :n_clusters], label)
            scores_km.append(testScore)
            print('Memory Usage: ', psutil.virtual_memory().percent)

            print('Saving Results')
            domains = label
            nc = str(n_range[n])
            writePDB(pdb + '_' + nc + '_domains.pdb', calphas, beta=domains)

        return labels, scores_km

    from sklearn.preprocessing import normalize
    print(os.getcwd())
    os.chdir("../results/subdivisions/")
    if not os.path.exists(pdb):
        os.mkdir(pdb)
    os.chdir(pdb)

    print('Spectral Clustering')
    n_range = np.arange(cluster_start, cluster_stop, cluster_step)
    n_evecs = max(n_range)
    start = time.time()
    maps = embedding(n_evecs, sims)
    end = time.time()
    print(end - start, ' Seconds')
    normalize(maps, copy=False)
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
