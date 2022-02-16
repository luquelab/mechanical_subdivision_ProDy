import numpy as np
def eigenCutoff(evals, threshold=0.001):
    sum = 0
    inv = 1/evals
    for i in range(len(evals)):
        sum += inv[i]
        contribution = inv[i]/sum
        if contribution < threshold and inv[i+1] < inv[i]:
            print('Eigenvalues Chosen up to: ', i)
            print(contribution)
            return i
    print('All Eigs Used')
    print(contribution)
    return i