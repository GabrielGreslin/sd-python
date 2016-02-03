__author__ = 'Gabriel'

'''
Basic test of the adaBoost algorithm on opt recognition
'''


import opticalCharacterManipulation
import sklearn.ensemble as ada
import time

(X,Y,X_test,Y_test) = opticalCharacterManipulation.loadTrainAndTestRawData()

start = time.time()

# Generate model
classifier = ada.AdaBoostClassifier()#{True:"balanced",False:"balanced"})
classifier.fit(X, Y)
#pred = classifier.predict(X_test)
score = classifier.score(X_test,Y_test)
print(score)
print("Elapsed time : "+str(time.time() - start) )

#0.565943238731
#Elapsed time : 0.5740330219268799