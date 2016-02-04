__author__ = 'Gabriel'

from sklearn.ensemble import RandomForestClassifier as rf
import opticalCharacterManipulation
import time

(X,Y,X_test,Y_test) = opticalCharacterManipulation.loadTrainAndTestRawData()


start = time.time()

# Generate model
classifier = rf()
classifier.fit(X, Y)

score = classifier.score(X_test,Y_test)
print(score)
print("Elapsed time : "+str(time.time() - start) )

