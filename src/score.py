import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from sklearn.metrics import pairwise_distances

@nb.njit()
def calcCentroids(X, labels, n_clusters):
    centroids = np.zeros((labels.shape[0], n_clusters))
    n = n_clusters
    for i in range(n_clusters):
        mask = (labels==i)
        if not np.any(mask):
            n += -1
            centroids[i,:] = np.random.rand(n_clusters)
        else:
            clust = X[mask,:]
            cent = np.mean(clust, axis=0)
            centroids[i,:] = cent

    if n != n_clusters:
        print('Some clusters unassigned')
        print('Assigned Clusters: ', n)

    return np.array(centroids)

@nb.njit()
def calcCosCentroids(X, labels, n_clusters):
    centroids = np.zeros((n_clusters, n_clusters))
    n = n_clusters
    for i in range(n_clusters):
        mask = (labels==i)
        if not np.any(mask):
            n += -1
            centroids[i,:] = np.random.rand(n_clusters)
        else:
            clust = X[mask,:]
            c = np.sum(clust, axis=0)
            cent = c/np.linalg.norm(c)
            centroids[i,:] = cent

    if n != n_clusters:
        print('Some clusters unassigned')
        print('Assigned Clusters: ', n)

    return centroids


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
    from sklearn.preprocessing import normalize

    # n_rand = 10
    # for i in range(n_rand):
    #     randcoords = np.random.normal(size=coords.shape)
    #     randcoords = normalize(randcoords)
    #     dists = pairwise_distances(randcoords, centroids, metric='cosine')
    #     d2min = np.partition(dists, kth=2)[:, :2]
    #     if scoreMethod == 'median':
    #         b = np.median(d2min[:, 1])
    #         a = np.median(d2min[:, 0])
    #     else:
    #         b = np.mean(d2min[:, 1])
    #         a = np.mean(d2min[:, 0])
    #     r_score = b / a

    dists = pairwise_distances(coords,centroids, metric='cosine')
    cdist = pairwise_distances(centroids, centroids, metric='cosine')
    normal = cdist.mean()
    d2min = np.partition(dists, kth=2)[:,:2]

    a = d2min[:,1]
    b = d2min[:,0]
    s = a/b

    if scoreMethod == 'median':
        score = np.median(s)
    else:
        score = np.mean(s)

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

    #_, _, title = getPDB(pdb)
    title = pdb

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

def volFlucts(coords, evals, evecs, gamma):
    from scipy.spatial import ConvexHull

    evals = evals*gamma

    print(coords.shape)
    n_atoms = coords.shape[0]
    #centroid = coords.mean(axis=0)
    print(evecs.shape)
    #coords += centroid
    hull = ConvexHull(coords)
    vol = hull.volume
    print('vol', vol)
    v_rms = 0
    vplot = np.zeros(len(evals))
    vmodes = np.zeros(coords.shape[0])
    h = 1
    for i in range(len(evals)):
        vec = evecs[:,i].reshape(-1,3)
        c1 = coords + h*vec
        c2 = coords - h*vec
        c3 = coords + 2*h * vec
        c4 = coords - 2*h * vec
        v1 = ConvexHull(c1).volume
        v2 = ConvexHull(c2).volume
        v3 = ConvexHull(c3).volume
        v4 = ConvexHull(c4).volume
        dv = (-v3 + 8*v1 - 8*v2 + v4)/(12*h)
        #dv = (v1 - v2)/(2*h)
        dvo = dv**2/evals[i]
        vplot[i] = dvo
        v_rms += dvo
        vmodes += np.sum(vec*dvo, axis=1)

    compressibility = v_rms/vol

    print(np.max(vplot))
    print(np.argmax(vplot))

    return compressibility, v_rms, vplot, vmodes

def solventVol(coords, radius):
    n = coords.shape[0]*3
    radii = [radius]*n
    from freesasa import calcCoord
    test = calcCoord(coords.flatten(), radii)

@nb.njit()
def sphereEigs(coords, evecs):
    ne = evecs.shape[-1]
    na = coords.shape[0]
    sevecs = np.empty((na, 3, ne))
    evecs = evecs.reshape((na, 3, ne))
    for i in range(ne):
        for j in range(na):
            v = evecs[j,:,i]
            pos = coords[j,:]
            sevecs[j,:,i] = sphereDelta(pos, v)

    return sevecs

@nb.njit()
def coordToSphere(coords):
    scoords = np.empty_like(coords)
    n = coords.shape[0]
    for i in nb.prange(n):
        scoords[i,:] = cartToSphere(coords[i,:])

@nb.njit()
def sphereDelta(pos, vec):
    return cartToSphere(pos+vec) - cartToSphere(pos)

@nb.njit()
def cartToSphere(vec):
    r = np.linalg.norm(vec)
    az = np.arctan2(vec[0],vec[1])
    pol = np.arccos(vec[2]/r)
    return np.array([r, az, pol])


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

@nb.njit()
def corrDiags(evals, evecs, nev, nnodes):
    diags = np.zeros((nnodes, 3, 3))
    for i in range(nev):
        eve = evecs[:,i]
        ev = 1/evals[i]
        for j in range(nnodes):
            vec = eve[3*j:3*j+3]
            diags[j,:,:] += ev * np.outer(vec,vec)
    return diags

@nb.njit()
def scorrDiags(evals, evecs, nev, nnodes, coords):
    diags = np.zeros((nnodes, 3, 3))
    for i in range(nev):
        eve = evecs[:,i]
        ev = 1/evals[i]
        for j in range(nnodes):
            vec = eve[3*j:3*j+3]
            c = coords[j]
            svec = sphereDelta(c, vec)
            diags[j,:,:] += ev * np.outer(svec,svec)
    return diags


#@nb.njit()
def residueCorrEigs(evals, evecs):
    print(evecs.shape)
    nnodes = int(evecs.shape[0]/3)
    nev = evals.shape[0]
    cDiags = corrDiags(evals, evecs, nev, nnodes)
    pVecs = np.zeros((nnodes, 3, 3))
    sVals = np.zeros((nnodes,3))
    for i in range(nnodes):
        mat = cDiags[i,:,:]
        u, s, v = np.linalg.svd(mat)
        #sval, sind = (sv.max(), sv.argmax())
        svec = u #[sind,:]
        sVals[i,:] = s
        pVecs[i,:,:] = svec

    return sVals, pVecs

def corrStrains(coords, evals, evecs, pdb):
    sVals, pVecs = residueCorrEigs(evals, evecs)
    np.savez_compressed(pdb + 'pvecs.npz', svals = sVals, pvecs = pVecs, coords=coords)
    return sVals, pVecs

#@nb.njit()
def sphereTransformMat(vec):
    svec = cartToSphere(vec)
    r, phi, theta = svec

    st = np.sin(theta)
    sp = np.sin(phi)
    ct = np.cos(theta)
    cp = np.cos(phi)

    mat = np.array([[st*cp,st*sp,ct],[ct*cp,ct*sp, -st],[-sp, cp, 0]])

    print(mat)
    print(svec, (mat @ (vec)/r)*r)
    print(np.allclose(svec, mat @ vec))

    return mat

#@nb.njit()
def sphereCrossCorr(cDiags, coords):
    n = coords.shape[0]
    sCCDiags = np.zeros((n,3,3))

    for i in range(n):
        diag = cDiags[i]
        vec = coords[i]
        mat = sphereTransformMat(vec)
        sCCDiags[i] = mat @ diag @ mat.T

    return sCCDiags

def sphereCC(coords, evals, evecs, pdb):
    nnodes = int(evecs.shape[0] / 3)
    nev = evals.shape[0]
    scDiags = scorrDiags(evals, evecs, nev, nnodes, coords)
    #cDiags = corrDiags(evals, evecs, nev, nnodes)
    #sCCDiags = sphereCrossCorr(cDiags, coords)
    rFlucts = scDiags[:,0,0]
    tFlucts = scDiags[:,1,1]
    pFlucts = scDiags[:,2,2]

    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 5))
    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': 11}
    matplotlib.rc('font', **font)

    n_asym = int(nnodes / 60)
    np.savez_compressed('../results/subdivisions/' + pdb + '_sCC.npz', scc = scDiags, rf = rFlucts)
    ax[0].plot(np.arange(rFlucts.shape[0])[:int(n_asym)], rFlucts[:int(n_asym)], label='Radial Fluctuations')
    ax[0].set_ylabel(r'$Å^{2}$', fontsize=12)
    ax[1].plot(np.arange(tFlucts.shape[0])[:int(n_asym)], tFlucts[:int(n_asym)], label='Azimuthal Fluctuations')
    ax[2].plot(np.arange(pFlucts.shape[0])[:int(n_asym)], pFlucts[:int(n_asym)], label='Polar Fluctuations')
    ax[2].set_xlabel('Residue Number', fontsize=12)
    ax[2].tick_params(axis='y', labelsize=8)
    ax[2].tick_params(axis='x', labelsize=8)

    #ax.legend()
    fig.suptitle(
        'Radial Fluctuations: ' + ' (' + pdb + ')', fontsize=12)

    plt.savefig('../results/subdivisions/' + pdb + '_rFlucts.svg')
    plt.savefig('../results/subdivisions/' + pdb + '_rFlucts.png')
    plt.show()
