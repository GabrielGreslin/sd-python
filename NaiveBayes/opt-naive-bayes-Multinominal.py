

__author__ = 'Gabriel'
'''
Basic test of the naive bayesiens algorithm on opt recognition
'''

import opticalCharacterManipulation
from sklearn.naive_bayes import MultinomialNB
import time

(X,Y,X_test,Y_test) = opticalCharacterManipulation.loadTrainAndTestRawData()

start = time.time()
print(X)
# Generate model
gnb = MultinomialNB()
gnb.fit(X, Y)
pred = gnb.predict(X_test)

s = gnb.score(X_test, Y_test);
print("Score:" + str(s))

end = time.time()
print("Time elapsed : " + str(end-start))

#Score:0.889259877574
#Time elapsed : 0.5560309886932373