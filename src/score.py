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


# def discretize_score(coords, labels):
#     for i in range(n_clusters):
#         mask = (labels==i)
#         if not np.any(mask):
#             print('Some clusters unassigned')
#
#         else:
#             clust = X[mask,:]
#             cent = np.mean(clust, axis=0)
#             centroids.append(cent)

def median_score(coords, centroids):
    from settings import scoreMethod
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
    from make_model import getPDB

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 10}

    _, _, title = getPDB(pdb)

    matplotlib.rc('font', **font)

    scores = []
    vars = []
    ntypes = []
    inerts = []
    for i in range(len(n_range)):
        nc = n_range[i]
        results = np.load('../results/subdivisions/' + pdb + '/' + pdb + '_' + str(nc) + '_results.npz')
        score = results['score']
        ntype = results['ntypes']
        var = results['var']
        inert = results['inertia']
        scores.append(score)
        vars.append(var)
        ntypes.append(ntype)
        inerts.append(inert)
    scores = np.array(scores)
    vars = np.array(vars)
    ntypes = np.array(ntypes)
    print('Plotting')
    fig, ax = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
    fig.suptitle('k profile: ' + title.title() + ' (' + pdb + ')')
    ax[0].scatter(n_range, scores, marker='D', s=15)
    ax[0].plot(n_range, scores)
    ax[1].plot(n_range, ntypes)
    ax[1].scatter(n_range, ntypes, marker='D', s=15)
    ax[2].scatter(n_range, inerts)
    ax[2].plot(n_range, inerts)
    ax[0].axvline(x=n_range[np.argmax(scores)], label='Best Score = ' + str(n_range[np.argmax(scores)]) , color='black')
    ax[1].axvline(x=n_range[np.argmax(scores)], label='Best Score', color='black')
    ax[2].axvline(x=n_range[np.argmax(scores)], label='Best Score', color='black')
    nc = str(n_range[np.argmax(scores)])
    # ax[0].set_xticks(n_range)
    ticks = ax[0].get_xticks()
    ticks = np.append(ticks, n_range[np.argmax(scores)])
    ax[2].set_xticks(ticks)
    ax[2].set_xlim([0, n_range[-1]])
    ax[2].set_xlabel('# Of Clusters')
    ax[0].set_ylabel('Quality' + '\n' + 'Score', rotation='horizontal', ha='center', va='center', labelpad=25)
    ax[1].set_ylabel('# Unique \n Clusters', rotation='horizontal', ha='center', va='center', labelpad=25)
    ax[2].set_ylabel('Intra \n Cluster \n Fluctuations', rotation='horizontal', ha='center', va='center', labelpad=25)

    ax[0].tick_params(axis='y', labelsize=8)
    ax[1].tick_params(axis='y', labelsize=8)
    ax[2].tick_params(axis='y', labelsize=8)

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend()
    # fig.tight_layout()
    print(pdb + '_' + nc + '_domains.png')
    if save:
        plt.savefig('../results/subdivisions/' + pdb + '_' + nc + '_domains.svg')
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
    # print('Collectivity Of Motion', kc)
    return kc

def meanStiff(evals):
    return 1/np.sum(1/evals[:100])

def meanCollect(evecs, evals, bfactors):
    n = evals.shape[0]
    meank = 0
    for i in range(n):
        mode = evecs[:,i]
        sqFlucts = mode**2

        if sqFlucts.shape[0] != bfactors.shape[0]:
            sqFlucts = np.reshape(sqFlucts, (-1, 3)).sum(axis=-1)

        kc = collectivity(sqFlucts)
        meank += 1/evals[i]*kc
    meank = meank/n
    print('Mean Collectivity Of Motion', meank)
    return meank

def forcedDisplacement(evals, evecs, forcevec):
    n = evals.shape[0]
    displacement = np.zeros_like(forcevec)
    for i in range(n):
        mode = evecs[:, i]
        displacement += 1/evals[i] * mode * mode.dot(forcevec)

    return displacement

def effectiveSpringConstant(coords, evals, evecs):
    from scipy.spatial import ConvexHull
    n = evals.shape[0]
    print(coords.shape)
    n_atoms = coords.shape[0]
    centroid = coords.mean(axis=0)
    print(centroid.shape)
    coords -= centroid
    baseRads = np.linalg.norm(coords, axis=1)
    lens = np.linalg.norm(coords, axis=1)
    norms = coords * 1 / lens[:, np.newaxis]
    ks = meanK(evecs, evals, norms)

    return ks

def meanK(ev, evals, d):
    n_e = evals.shape[0]
    ev1 = ev * 1 / np.sqrt(evals)
    ev2 = ev * np.sqrt(evals)
    ev1 = np.reshape(ev1, (-1, 3, n_e))
    ev2 = np.reshape(ev2, (-1, 3, n_e))
    top = np.zeros(ev.shape[0])
    bot = np.zeros(ev.shape[0])
    dot1 = np.sum(np.sum(np.abs(ev1*d[:,:,np.newaxis]), axis=1), axis=-1)
    dot2 = np.sum(np.sum(np.abs(ev2 * d[:, :, np.newaxis]), axis=1), axis=-1)
    print(dot2)
    return dot1/dot2

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
    vol1 = hull.volume
    print('vol', vol1)
    lens = np.linalg.norm(coords, axis=1)
    # rAvg = np.mean(lens)

    hess = gamma*hess
    vs = [0]
    volumes = [0]
    for i in range(200):
        norms = coords * 1/lens[:,np.newaxis]
        ncoords = ((i-100+1)/500)*norms
        vec = ncoords.flatten()
        v = 1/2 * np.dot(vec.T, hess.dot(vec))
        volcoords = coords + ncoords
        hull = ConvexHull(volcoords)
        vol = hull.volume
        vs.append(v)
        volumes.append(vol - vol1)
    a = fitBulk(np.array(volumes), vs)
    bulkmod = vol1*a
    line = parabola(np.array(volumes), a)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(volumes, vs, label='volume vs pressure')
    ax.plot(volumes, line, label='volume vs pressure')
    ax.legend()
    plt.show()

    return bulkmod

def parabola(x, b):
    return 1/2*b*x**2

def fitBulk(volumes, energies):
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(parabola, volumes, energies)
    print(popt)
    return popt

def stresses(hess, distFlucts):
    n_atoms = distFlucts.shape[0]
    # real stresses can be sum over distance outer products weighted by distance fluctuations
    hess = hess.tobsr(blocksize=(3,3))
    resStress = np.zeros((n_atoms,3,3))
    for i in range(3):
        drow = distFlucts.data[distFlucts.indptr[i]:distFlucts.indptr[i+1]]
        hrow = hess.data[hess.indptr[i]:hess.indptr[i+1]]
        diagColumn = hess.indices[hess.indptr[i]:hess.indptr[i+1]]
        dInd = np.argwhere(diagColumn==i)
        hrow[dInd[0]] = np.zeros((3,3))
        resStress[i] = hrow.sum(axis=0)

    return resStress

    # print('block row', brow.sum(axis=0))
    #stress = hess.sum(axis=1)
    #print('stress shape', stress.shape)

def overlapStiffness(evals, evecs, coords):
    n = evals.shape[0]
    print(coords.shape)
    n_atoms = coords.shape[0]
    centroid = coords.mean(axis=0)
    print(centroid.shape)
    coords -= centroid
    lens = np.linalg.norm(coords, axis=1)
    norms = coords * 1 / lens[:, np.newaxis]
    norms = norms.flatten()
    k = 0
    d = 0
    for i in range(len(evals)):
        e = evals[i]
        vec = evecs[:,i]

        overlap = vec.dot(norms)/(np.linalg.norm(vec)*np.linalg.norm(norms))
        d += np.sqrt(overlap**2)
        k += np.sqrt(e)*overlap

    return k