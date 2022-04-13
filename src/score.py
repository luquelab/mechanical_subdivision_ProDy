import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances

def calcCentroids(X, labels, n_clusters):
    centroids = []
    for i in range(n_clusters):
        mask = (labels==i)
        if not np.any(mask):
            print('Some clusters unassigned')
            centroids.append((np.random.rand(n_clusters)))
            #return np.array(centroids), True
        else:
            clust = X[mask,:]
            cent = np.mean(clust, axis=0)
            centroids.append(cent)



    return np.array(centroids), False


def discretize_score(coords, labels):
    for i in range(n_clusters):
        mask = (labels==i)
        if not np.any(mask):
            print('Some clusters unassigned')

        else:
            clust = X[mask,:]
            cent = np.mean(clust, axis=0)
            centroids.append(cent)

def median_score(coords, centroids):
    from input import scoreMethod
    from sklearn.metrics import pairwise_distances

    dists = pairwise_distances(coords,centroids, metric='cosine')
    cdist = pairwise_distances(centroids, centroids, metric='cosine')
    normal = cdist.mean()
    d2min = np.partition(dists, kth=2)[:,:2]

    if scoreMethod == 'median':
        b = np.median(d2min[:,1])
        a = np.median(d2min[:,0])
    else:
        b = np.mean(d2min[:, 1])
        a = np.mean(d2min[:, 0])
    score = b/a
    return score

def cluster_types(labels):
    _, counts = np.unique(labels, return_counts=True)
    thresh = 0.05*np.mean(counts)
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
            'size': 24}

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

def clustFlucts(labels, pdb):
    data = np.load('../results/subdivisions/' + pdb + '_sqFlucts.npz')
    sqFlucts = data['sqFlucts']
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    n_clusters = np.max(labels)
    for i in range(2):
        mask = (labels==i)
        clustFl = sqFlucts[mask]
        print(clustFl.shape)
        ax.plot(np.arange(clustFl.shape[0]), clustFl, label=str(i))
    plt.show()


def globalPressure(coords, hess, gamma):
    from scipy import sparse
    from scipy.spatial import ConvexHull
    # center coords

    print(coords.shape)
    n_atoms = coords.shape[0]
    centroid = coords.mean(axis=0)
    print(centroid.shape)
    coords += centroid
    hull = ConvexHull(coords)
    vol = hull.volume
    lens = np.linalg.norm(coords, axis=1)
    # rAvg = np.mean(lens)

    hess = gamma*hess
    vs = [0]
    volumes = [vol]
    for i in range(200):
        norms = coords * 1/lens[:,np.newaxis]
        ncoords = ((i-100+1)/50)*norms
        vec = ncoords.flatten()
        v = 1/2 * np.dot(vec.T, hess.dot(vec))
        volcoords = coords + ncoords
        hull = ConvexHull(volcoords)
        vol = hull.volume
        vs.append(v)
        volumes.append(vol)
    print(vs)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(volumes, vs, label='volume vs pressure')
    ax.legend()
    plt.show()

# def vStress(i,j, hess, evec):
#     vec =
#
# def virialStress(vec, coords, hess):

def forceVec(evec, hess):
    forces = hess.dot(evec)
    forces = np.reshape(forces,(-1,3))
    magnitudes = np.linalg.norm(forces, axis=1)
    return forces, magnitudes

def collectivity(sqFlucts):
    n = sqFlucts.shape[0]
    alpha = 1/sqFlucts.sum()
    k = 0
    for i in range(n):
        k += alpha*sqFlucts[i] * np.log(alpha * sqFlucts[i])
    kc = np.exp(-k)
    kc = kc/n
    print('Collectivity Of Motion', kc)
    return k