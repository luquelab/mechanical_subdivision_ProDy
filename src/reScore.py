import numpy as np
from sklearn.preprocessing import normalize
from score import *

pdb = '1ohg'
n_range = np.arange(72,422,2)

scores = []
vars = []
ntypes = []
n_range = np.arange(72,422,2)
maps = np.load('../results/models/' + pdb + 'embedding.npy')
for i in range(len(n_range)):
    n_clusters = n_range[i]
    emb = maps[:, :n_clusters]
    normalize(emb, copy=False)
    results = np.load('../results/subdivisions/' + pdb + '/' + pdb + '_' + str(n_clusters) + '_results.npz')
    labels = results['labels']

    centroids = calcCentroids(emb, labels, n_clusters)
    testScore = median_score(emb, centroids)
    var, ntype = cluster_types(labels)

    np.savez('../results/subdivisions/' + pdb + '/' + pdb + '_' + str(n_clusters) + '_results', labels=labels, score=testScore,
             var=var, ntypes=ntype, n=results['n'], method=results['method'])


