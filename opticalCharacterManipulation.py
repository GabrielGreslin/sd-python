__author__ = 'Gabriel'

import numpy as np
import os,sys


'''
Load the data in a X and Y matrice, without any normalization
'''
def builtXndY(data):
    samples_number = len(data)
    feature_number = 64
    X = np.zeros((samples_number,feature_number),dtype=float)
    Y = np.zeros((samples_number,1),dtype=float)

    for i,l in enumerate(data):
        X[i,:] = (l[:-1]) # normalization
        Y[i,0]= l[-1]

    return (X,Y)

def loadTrainAndTestRawData():

    dir = os.path.dirname(__file__)
    print()
    filenameTrain = os.path.realpath("{0}\\data\\Optical character recognition\\optdigits.tra".format(dir))
    filenameTest = os.path.realpath("{0}\\data\\Optical character recognition\\optdigits.tes".format(dir))

    data = np.genfromtxt(filenameTrain, delimiter=',', dtype=None,skip_header=0)
    print("Data Train Shape :" + str(data.shape))
    datatest = np.genfromtxt(filenameTest, delimiter=',', dtype=None,skip_header=0)
    print("Data Test Shape :" + str(data.shape))

    (X,Y)= builtXndY(data)
    (X_test,Y_test)= builtXndY(datatest)

    return (X,Y,X_test,Y_test)



