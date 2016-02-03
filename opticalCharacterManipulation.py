__author__ = 'Gabriel'

import numpy as np

'''
Load the data in a X and Y matrice, normalized
'''
def builtXndY(data):
    samples_number = len(data)
    feature_number = 64
    X = np.zeros((samples_number,feature_number),dtype=float)
    Y = np.zeros((samples_number,1),dtype=float)

    for i,l in enumerate(data):
        X[i,:] = (l[:-1]-8)/16 # normalization
        Y[i,0]= l[-1]

    return (X,Y)