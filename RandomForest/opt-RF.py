__author__ = 'Gabriel'

import sys
sys.path.append('../')

from sklearn.ensemble import RandomForestClassifier as rf
import opticalCharacterManipulation
import time
from features import *
import numpy

(X,Y,X_test,Y_test) = opticalCharacterManipulation.loadTrainAndTestFeaturesData(sidePoints)

start = time.time()

# Generate model
classifier = rf()
classifier.fit(X, numpy.ravel(Y))

score = classifier.score(X_test,Y_test)
print(score)
print("Elapsed time : "+str(time.time() - start) )
