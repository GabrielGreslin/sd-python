# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:13:19 2016

@author: Abderrahmen
"""

# Import
from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt

# Import spam data
##################

# Load word stats
word_stats = genfromtxt('data\Spam\emails-train-features.txt', delimiter=' ', dtype='i,i,i')
print('word_stats:')
print(word_stats)

# Load label stats
labels = genfromtxt('data\Spam\emails-train-labels.txt', delimiter=' ', dtype='i')
print('labels:')
print(labels)

# Generate model
gnb = GaussianNB()
model = gnb.fit(word_stats, labels)

# Predict label (spam/no spam)
#y_pred = model.predict(iris.data)

# Print model results
#print('iris.data:')
#print(iris.data)

#print('iris.target:')
#print(iris.target)

#print('y_pred:')
#print(y_pred)