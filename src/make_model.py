from prody import *
import sys
import os
#import requests
#import gzip

pdb = sys.argv[1]
filename = pdb + '.pdb'
n_modes = int(sys.argv[2])

print(os.getcwd())
os.chdir("../data/asym_pdbs/")
print(os.getcwd())

# if not os.path.exists(filename):
#     print('downloading')
#     vdb_url = 'http://viperdb.scripps.edu/resources/OLIGOMERS/1a34_full.vdb.gz'
#     handle = requests.get(vdb_url, allow_redirects=True)
#     #open(filename,'wb').write(handle.content)
#     decompressedFile = gzip.open(fileobj=handle.content)
#     with open(filename, 'wb') as outfile:
#         outfile.write(decompressedFile.read())


capsid = parsePDB(filename)
calphas = capsid.select('calpha').copy()
gnm = GNM(pdb + '_full')
gnm.buildKirchhoff(calphas, cutoff=10.0, kdtree=True, sparse=True)

gnm.calcModes(n_modes,turbo=True)

print(os.getcwd())
os.chdir("../../results/models")
print(os.getcwd())
saveModel(gnm,matrices=True)
saveAtoms(calphas,filename='calphas_' + pdb)
