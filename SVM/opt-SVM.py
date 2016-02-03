__author__ = 'Gabriel'

'''
Basic test of the SVM algorithm on opt recognition
'''


import opticalCharacterManipulation
from sklearn import svm
import time

(X,Y,X_test,Y_test) = opticalCharacterManipulation.loadTrainAndTestRawData()

start = time.time()

# Generate model
classifier = svm.SVC(class_weight="balanced",C=2,kernel = 'rbf')
classifier.fit(X, Y)
score = classifier.score(X_test,Y_test)
print(score)
print("Elapsed time : "+str(time.time() - start) )
# Score :0.58653311074
# Elapsed time : 3.7342140674591064