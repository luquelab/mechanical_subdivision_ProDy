
def nmdWrite(pdb, coords, modes, sqFlucts):
    print('WRITING NMD')
    import numpy as np
    nodes = coords.shape[0]
    nmodes = modes.shape[1]
    filename = 'C:/Users/colin/OneDrive - San Diego State University (SDSU.EDU)/Research/Domain_Subdivision/mechanical_subdivision_ProDy/src/' + pdb + '_ico.nmd'
    np.set_printoptions(suppress=True)
    print(coords.shape)
    with open(filename, 'w') as f:
        # f.write('nmwiz_load ' + filename)
        # f.write('\n')
        f.write('name ' + str(nmodes) + '_modes_' + pdb)
        f.write('\n')

        f.write('bfactors ')
        # c = np.array2string(sqFlucts.flatten(), threshold=3*nodes+100, separator=' ', precision=8, formatter={'float_kind':lambda x: "%.8f" % x}).replace('\n', '').replace('[','').replace(']','')
        for c in sqFlucts.flatten():
            f.write('%.3f' % c + ' ')
        f.write('\n')

        f.write('resids')
        for i in range(int(nodes/3)):
            f.write(' ' + str(100 +i %200))
        f.write('\n')

        f.write('coordinates ')
        coords.tofile(f, ' ', '%.3f')
        f.write('\n')
        for i in range(nmodes):
            f.write('mode ' + str(i) + ' ')
            mode = np.abs(modes[:,i])
            # c = np.array2string(mode.flatten(), threshold=3*nodes+100, separator=' ', precision=8, formatter={'float_kind':lambda x: "%.8f" % x}).replace('\n', '').replace('[','').replace(']','')
            for c in mode.flatten():
                f.write('%.3f' % c + ' ')
            f.write('\n')
        f.close()
