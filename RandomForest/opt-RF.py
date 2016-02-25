__author__ = 'Gabriel'

import sys
sys.path.append('../')

from sklearn.ensemble import RandomForestClassifier as rf
import opticalCharacterManipulation
import time
from features import *
import numpy

(X,Y,X_test,Y_test) = opticalCharacterManipulation.loadTrainAndTestRawData()
# (X,Y,X_test,Y_test) = opticalCharacterManipulation.loadTrainAndTestFeaturesData(sidePoints)
# (X,Y,X_test,Y_test) = opticalCharacterManipulation.loadTrainAndTestFeaturesData(sidePoints, gravityPoints)

start = time.time()

# Generate model
N = 200
score = 0
for i in range(0, N) :
    classifier = rf()
    classifier.fit(X, numpy.ravel(Y))
    score += classifier.score(X_test,Y_test)
score = round(100 * score / N, 2)
print("Score: " + str(score) + " %")
print("Elapsed time : " + str(time.time() - start))
