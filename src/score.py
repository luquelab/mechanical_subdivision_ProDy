import numpy as np
from sklearn.metrics import pairwise_distances

def calcCentroids(X, labels, n_clusters):
    centroids = []
    for i in range(n_clusters):
        mask = (labels==i)
        if not np.any(mask):
            print('Some clusters unassigned')
            centroids.append((np.random.rand(n_clusters)))
        else:
            clust = X[mask,:]
            cent = np.mean(clust, axis=0)
            centroids.append(cent)

    return np.array(centroids)



def median_score(coords, centroids):
    from input import scoreMethod
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(coords,centroids)
    d2min = np.partition(dists, kth=2)[:,:2]
    if scoreMethod is 'median':
        b = np.median(d2min[:,1])
        a = np.median(d2min[:,0])
    else:
        b = np.mean(d2min[:, 1])
        a = np.mean(d2min[:, 0])
    score = b/a
    return score

def cluster_types(labels):
    thresh = 5
    _, counts = np.unique(labels, return_counts=True)
    counts = np.rint(counts/thresh)*thresh
    var = np.std(counts)
    ntypes = np.unique(counts).shape[0]
    return var, ntypes

def plotScores(pdb, n_range, save=False):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    scores = []
    vars = []
    ntypes = []
    for i in range(len(n_range)):
        nc = n_range[i]
        results = np.load('../results/subdivisions/' + pdb + '/' + pdb + '_' + str(nc) + '_results.npz')
        score = results['score']
        ntype = results['ntypes']
        var = results['var']
        scores.append(score)
        vars.append(var)
        ntypes.append(ntype)
    scores = np.array(scores)
    vars = np.array(vars)
    ntypes = np.array(ntypes)
    print('Plotting')
    fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(pdb)
    ax[0].scatter(n_range, scores, marker='D')
    ax[0].plot(n_range, scores)
    ax[1].plot(n_range, ntypes)
    ax[1].scatter(n_range, ntypes)
    ax[1].set_ylabel('# Of Unique Clusters')
    ax[0].axvline(x=n_range[np.argmax(scores)], label='Best Score' , color='black')
    ax[1].axvline(x=n_range[np.argmax(scores)], label='Best Score', color='black')
    nc = str(n_range[np.argmax(scores)])
    # ax[0].set_xticks(n_range)
    ticks = ax[1].get_xticks()
    ticks = np.append(ticks, n_range[np.argmax(scores)])
    ax[1].set_xticks(ticks)
    ax[1].set_xlabel('# Of Clusters')
    ax[0].set_ylabel('Quality Score')
    ax[0].legend()
    fig.tight_layout()
    print(pdb + '_' + nc + '_domains.png')
    if save:
        plt.savefig('../results/subdivisions/' + pdb + '_' + nc + '_domains.png')
    plt.show()
