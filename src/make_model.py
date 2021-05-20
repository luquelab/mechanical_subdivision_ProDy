from prody import *
import sys
import os
import wget
import shutil
import gzip
#import requests
#import gzip


def make_model(pdb, n_modes):
    filename = pdb + '_full.pdb'


    os.chdir("../data/capsid_pdbs/")

    if not os.path.exists(filename):
        vdb_url = 'http://viperdb.scripps.edu/resources/OLIGOMERS/' + pdb + '_full.vdb.gz'
        print(vdb_url)
        vdb_filename = wget.download(vdb_url)
        with gzip.open(vdb_filename, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


    capsid = parsePDB(filename)
    calphas = capsid.select('ca').copy()
    gnm = GNM(pdb + '_full')
    gnm.buildKirchhoff(calphas, cutoff=10.0, kdtree=True, sparse=True)
    gnm.calcModes(n_modes,turbo=True)

    print(os.getcwd())
    os.chdir("../../results/models")
    print(os.getcwd())
    saveModel(gnm,matrices=True)
    saveAtoms(calphas,filename='calphas_' + pdb)

    return gnm, calphas
