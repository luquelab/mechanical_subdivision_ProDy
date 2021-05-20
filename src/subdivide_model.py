



def subdivide_model(pdb, n_clusters, gnm_in = None ,calphas_in=None):
    print(os.getcwd())
    if gnm_in is None:
        gnm = loadModel(pdb + '_full.gnm.npz')
    else:
        gnm = gnm_in
    if calphas_in is None:
        calphas = loadAtoms('calphas_' + pdb + '.ag.npz')
    else:
        gnm = gnm_in


    test = cluster.SpectralClustering(assign_labels='kmeans',n_clusters=n_clusters,n_init=10,affinity='nearest_neighbors',n_jobs=8).fit(gnm.getEigvecs())

    domains = test.labels_
    print(len(np.unique(domains)))
    calphas.setData('domain',domains)
    os.chdir("../results/subdivisions")
    writePDB(pdb + '_domains.pdb',calphas,beta=domains)