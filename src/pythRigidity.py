import numpy as np

#pythran export pairDists(float[:], float[:])
def pairDists(evi, evj):
    d = np.sum((evi-evj)**2)
    return d



#pythran export clusterFlucts(float[:,:] or float[:,:,:], float[:])
def clusterFlucts(evecs, cFlucts):
    c_size = evecs.shape[0]
    flucts = np.zeros(c_size)
    for i in range(c_size):
        evi = evecs[i,:]
        for j in range(c_size):
            evj = evecs[j,:]
            flucts[i] += pairDists(evi, evj)

    return flucts/(2*c_size)

#pythran export fastFlucts(float[:,:], str)
def fastFlucts(evecs, model):
    if model=='anm':
        ev = evecs**2
        f = np.sum(ev.reshape(-1,3,ev.shape[1]), axis=1)
    else:
        f = evecs**2
    return np.sum(f,axis=1)


