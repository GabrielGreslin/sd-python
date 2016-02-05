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
        X[i,:] = (l[:-1])
        Y[i,0]= l[-1]

    return (X,Y)

'''
Give the raw data
X,X_test has 64 columns
Y,Y_test has 1 columns

Split input in 75% training set and 25% testing set

return X,Y,X_test,Y_test
'''
def loadTrainAndTestRawData():

    dir = os.path.dirname(__file__)
    print()
    filenameTrain = os.path.realpath("{0}\\data\\Optical character recognition\\optdigits.tra".format(dir))
    filenameTest = os.path.realpath("{0}\\data\\Optical character recognition\\optdigits.tes".format(dir))

    data = np.genfromtxt(filenameTrain, delimiter=',', dtype=None,skip_header=0)
    print("Data Train Shape :" + str(data.shape))
    datatest = np.genfromtxt(filenameTest, delimiter=',', dtype=None,skip_header=0)
    print("Data Test Shape :" + str(datatest.shape))

    (X,Y)= builtXndY(data)
    (X_test,Y_test)= builtXndY(datatest)

    return (X,Y,X_test,Y_test)


'''
Give the raw data + additionnal features
X,X_test has 64 + #additionnalFeatures columns
Y,Y_test has 1 columns

Take a list of features function
A feature function is a function from R^64 -> R^n

Split input in 75% training set and 25% testing set

return X,Y,X_test,Y_test
'''
def loadTrainAndTestFeaturesData(*listFeatFunction):
    (X,Y,X_test,Y_test)= loadTrainAndTestRawData()

    N_F = getNumberArguments(*listFeatFunction)

    print("Total Number of new features : " + str(N_F))
    #Extend X
    (U,V) = X.shape
    X_feat = np.zeros((U,V+N_F))
    X_feat[:,:-N_F] = X

    #Extend X_test
    (U_test,V) = X_test.shape
    X_test_feat = np.zeros((U_test,V+N_F))
    X_test_feat[:,:-N_F] = X_test

    X_feat = appliedFeatures(X_feat,*listFeatFunction)
    X_test_feat = appliedFeatures(X_test_feat,*listFeatFunction)

    return  (X_feat,Y,X_test_feat,Y_test)

'''
Fill up the features
Assume the matrice X as the good size
'''
def appliedFeatures(X,*list_features):
        for lineNb,l in enumerate(X):
            column=64
            for f in list_features:
                addFeat = f(l[0:64])
                nbFeat = len(addFeat)
                X[lineNb,column:(column+nbFeat)] = addFeat
                column += nbFeat
        return X

'''
Compute the number of features creadted by the list of features function
'''
def getNumberArguments(*args):
    z = np.zeros(64)
    totalLength = 0

    for f in args:
        additionnalFeaturesNumber = len(f(z))
        totalLength += additionnalFeaturesNumber
        print(str(f.__name__) + " add "+str(additionnalFeaturesNumber)+" new features.")

    return totalLength



