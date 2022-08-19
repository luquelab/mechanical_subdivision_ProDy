import numpy as np
import numba as nb

def buildSymBlocks(hess, T):
    blocks, ris = extractInteractBlocks(hess, T)
    irreps = readIrreps('irreps.txt')
    symHessBlocks = irrepBuildFromBlocks(blocks, irreps, ris)
    return symHessBlocks, blocks


def extractInteractBlocks(mat, T):
    ncols = mat.shape[1]
    nblock = int(ncols/60)
    row = mat[:nblock,:]
    blocks = []
    ris = []
    for i in range(60):
        block = row[:, i*nblock:(i+1)*nblock]
        if block.nnz > 0:
            print(block.nnz)
            blocks.append(block)
            ris.append(i)

    blocks = localCoords(blocks, T)
    print(np.array(ris))
    return blocks, ris

def localCoords(blocks, rots):
    from scipy import sparse
    bs = []

    for i, b in enumerate(blocks):
        r = rots[i,:,:].copy()
        n = int(b.shape[0]/3)
        local = b.dot(sparse.kron(sparse.identity(n), r))
        bs.append(local)

    return bs

def irrepBuildFromBlocks(blocks, irreps, ris):
    from scipy import sparse
    nmat = blocks[0].shape[0]
    sizes = [1,3,3,4,5]
    symHessBlocks = []
    inds = [0,6,11,1,5]
    blocks = np.array(blocks)[[0,4,5,1,3]]

    for i, n in enumerate(sizes):
        sblock = sparse.lil_matrix((nmat*n,nmat*n))

        # for k,r in enumerate(inds):
        #     rotmat = irreps[r][i]
        #     print(blocks[k].shape, rotmat.shape)
        #     b = sparse.kron(rotmat, blocks[k])
        #     sblock += b
        #
        # symHessBlocks.append(sblock)

        for k in range(3):
            rotmat = irreps[inds[k]][i]
            #print(blocks[k].shape, rotmat.shape)
            #print(k)
            #print(rotmat)
            b = sparse.kron(rotmat, blocks[k])
            #print(b.shape, nmat*i)
            sblock += b

        b = sparse.lil_matrix((nmat*n,nmat*n))
        for k in range(3,5):
            rotmat = irreps[inds[k]][i]
            #print(k)
            #print(rotmat)
            b += sparse.kron(rotmat, blocks[k])

        sblock = sblock + b + b.T
        symHessBlocks.append(sblock)

    return symHessBlocks


def readIrreps(filename):
    count=0
    irreps = []
    with open(filename) as fp:
        Lines = fp.readlines()
        for line in Lines:
            if line.strip():
                if ':' in line:
                    if count >0:
                        irreps.append(repmats)
                    repmats = []
                    count +=1
                else:
                    if '[' in line:
                        mat = []
                    row = line.strip()
                    row = row.replace('[', '')
                    row = row.replace(']', '')
                    fltrow = [float(x) for x in row.split()]
                    mat.append(fltrow)
                    if ']' in line:
                        repmats.append(np.array(mat))
                    #print("Line{}: {}".format(count, repmats))
    return irreps

def readTfromFile(filename):
    T0 = np.zeros((60,60))
    i = 0
    with open(filename) as fp:
        Lines = fp.readlines()
        for line in Lines:
            if line.strip():
                row = line.strip()
                fltrow = [float(x) for x in row.split()]
                T0[i,:] = fltrow
                i += 1
    print(T0)
    return T0


def centerPlusRots(coords):
    from scipy.spatial.transform.rotation import Rotation as R
    import numpy as np
    centroid = np.mean(coords, axis=0)
    coords = coords - centroid
    n = int(coords.shape[0]/60)
    base = coords[:n,:]

    axes = np.zeros((60, 3))
    angles = np.zeros((60))
    dists = np.zeros((60))

    for i in range(60):
        asym = coords[i*n:(i+1)*n,:]
        rot, rmsd = R.align_vectors(base, asym)
        r = np.linalg.norm(np.mean(base,axis=0) - np.mean(asym,axis=0))
        vec = rot.as_rotvec()
        angle = np.linalg.norm(vec)
        axis = vec / angle
        axes[i] = axis
        angles[i] = angle
        dists[i] = r

    return coords, axes, angles, dists, base

def findAxes(angles, axes, dists):
    di = dists.argsort()
    angles = np.rad2deg(angles[di]).round()
    print(angles)
    axes = axes[di]
    ff1 = np.nonzero(angles == 72)
    ff2 = np.nonzero(angles==144)
    fax = np.concatenate((axes[ff1].copy()[:2], axes[ff2].copy()[:2]))
    print(ff1, ff2)

    for i in range(4):
        theta = np.dot(fax[0,:],fax[i,:])
        print(theta)
        if theta < 0:
            fax[i,:] = -fax[i,:]
    print(fax)
    five_fold_axis = np.mean(fax,axis=0)
    print(five_fold_axis)

    tf1 = np.nonzero(angles == 120)
    tax = axes[tf1].copy()[:2]
    print(tax)
    tax[0,:] = -tax[0,:]
    print(np.dot(tax[0],tax[1]))
    three_fold_axis = np.mean(tax, axis=0)
    print(three_fold_axis)

    print(np.rad2deg(np.arccos(np.dot(five_fold_axis, three_fold_axis))))

    return five_fold_axis, three_fold_axis

def buildFromAxes(acoords, ax5, ax3):
    from scipy.spatial.transform.rotation import Rotation as R
    r5 = [R.from_rotvec(np.array([0,0,0]))]
    units = [acoords]

    ax3s = [ax3]
    a = -2*np.pi/5

    for i in range(4):
        vec = ax5*(i+1)*a
        r = R.from_rotvec(vec)
        rcoords = r.apply(acoords)
        rax3 = r.apply(ax3)
        rax3 = rax3/np.linalg.norm(rax3)
        units.append(rcoords)
        ax3s.append(rax3)
        r5.append(r)

    print(ax3s)
    pcoords = np.vstack(units)
    a = -2 * np.pi / 3
    print(pcoords.shape)
    units = [pcoords]
    r3 = []
    for i in range(5):
        vec = ax3s[i]*a
        r = R.from_rotvec(vec)
        rcoords = r.apply(pcoords)
        units.append(rcoords)
        r3.append(r)

    halfcoords = np.vstack(units)
    print(halfcoords.shape)

    ax2_1 = (ax3s[2] + ax3s[3])/np.linalg.norm((ax3s[2] + ax3s[3]))
    r = R.from_rotvec(ax3s[0]*a)
    ax2_6 = r.apply(ax2_1)
    vec2 = -ax2_6*np.pi
    r = R.from_rotvec(vec2)
    hemicoords = r.apply(halfcoords)
    r2 = []
    r2.append(r)

    fullCoords = np.vstack((halfcoords, hemicoords))


    T = buildRfromRots(r5, r3, r2, fullCoords)

    print('done')
    return fullCoords, T

def buildRfromRots(r5, r3, r2, fullCoords):
    from scipy.spatial.transform.rotation import Rotation as Rot
    R = np.zeros((60,3,3))
    R[0,:,:] = np.identity(3)
    na = int(fullCoords.shape[0]/60)
    asym = fullCoords[:na,:]
    c = 1
    for i in range(4):
        r = r5[i+1].as_matrix()
        R[i+1,:,:] = r
        test = np.allclose(r.dot(asym.T).T, fullCoords[c*na:(c+1)*na])
        print(test)
        c += 1

    for i in range(5):
        for j in range(5):
            r33 = r3[i].as_matrix()
            r55 = r5[j].as_matrix()
            r =  r33.dot(r55)
            #print(r.as_matrix())
            test = np.allclose(r.dot(asym.T).T, fullCoords[c * na:(c + 1) * na])
            print(test)
            R[c,:,:] = r
            c += 1

    r2 = r2[0]

    for i in range(30):
        mat = R[i,:,:]
        r1 = Rot.from_matrix(mat)
        r = r1*r2
        R[30+i,:,:] = r.as_matrix()

    return R

def axisAngle(rot):
    import scipy.spatial.transform.rotation as R
    import numpy as np
    rot = R.Rotation.from_matrix(rot)
    vec = rot.as_rotvec()
    angle = np.linalg.norm(vec)
    axis = vec/angle
    return angle, axis





def symToLocal(evecs):
    print()
    return 0