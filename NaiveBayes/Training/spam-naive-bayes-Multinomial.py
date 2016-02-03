# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:13:19 2016

@author: Gabriel
"""

# Import
import numpy as np
from numpy import genfromtxt
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


# TODO: Use test data instead of building it myself

# Import spam data
##################

# Load word stats
word_stats = genfromtxt('data\Spam\emails-train-features.txt', delimiter=' ', dtype='i,i,i')

# Load label stats
y = genfromtxt('data\Spam\emails-train-labels.txt', delimiter=' ', dtype='i')

def fromWordStatsToX(word_stats):
    samples_size = max([x[0] for x in word_stats])
    dictionnary_size = max([x[1] for x in word_stats])

    X = np.zeros((samples_size,dictionnary_size+1),dtype=int)

    for x in word_stats:
        X[(x[0]-1),(x[1]-1)]=x[2]
        X[(x[0]-1),(x[1])]+=x[2]

    return X

X=fromWordStatsToX(word_stats)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=0)

# Generate model
gnb = MultinomialNB()
gnb.fit(X_train, y_train)
pred = gnb.predict(X_test)

s = gnb.score(X_test, y_test);
print("Score:")
print(s)

cr = classification_report(y_test,pred,target_names=["Not Spam","Is Spam"])
print(cr)

ac = confusion_matrix(y_test, pred)
print(ac)
'''
Cross validation not relevant as we don't want to fit the parameter model.
But we could fit the best alpha which is the additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing)
'''
'''
             precision    recall  f1-score   support

   Not Spam       0.97      0.98      0.97        86
    Is Spam       0.98      0.97      0.97        89

avg / total       0.97      0.97      0.97       175

[[84  2]
 [ 3 86]]

'''