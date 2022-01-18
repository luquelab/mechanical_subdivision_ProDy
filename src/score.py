import numpy as np
from sklearn.metrics import pairwise_distances


def median_score(coords, centroids):
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(coords,centroids)
    d2min = np.partition(dists, kth=2)[:,:2]
    score = np.mean(d2min[:,1]/d2min[:,0])
    return score

def cluster_types(labels):
    _, counts = np.unique(labels, return_counts=True)
    var = np.std(counts)
    ntypes = np.unique(counts).shape[0]
    return var, ntypes