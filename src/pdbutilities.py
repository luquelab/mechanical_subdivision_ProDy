import numpy as np
from prody import *

def extract_transforms(header):

    biomt = header.get('biomoltrans')
    keys = list(biomt)

    keys.sort()
    for i in keys:
        mt = biomt[i]
        # mt is a list, first item is list of chain identifiers
        # following items are lines corresponding to transformation
        # mt must have 3n + 1 lines
        if (len(mt)) % 4 != 0:
            LOGGER.warn('Biomolecular transformations {0} were not '
                        'applied'.format(i))
            continue

        T = []
        for times in range(int((len(mt)) / 4)):
            rotation = np.zeros((3, 3), dtype=np.float64)
            translation = np.zeros(3)
            line0 = np.fromstring(mt[times * 4 + 1], sep=' ')
            rotation[0, :] = line0[:3]
            translation[0] = line0[3]
            line1 = np.fromstring(mt[times * 4 + 2], sep=' ')
            rotation[1, :] = line1[:3]
            translation[1] = line1[3]
            line2 = np.fromstring(mt[times * 4 + 3], sep=' ')
            rotation[2, :] = line2[:3]
            translation[2] = line2[3]
            T.append(rotation)

        return np.array(T)

# def buildCapsidCustom(asym, header):



