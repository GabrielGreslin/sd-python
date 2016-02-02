# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:13:19 2016

@author: Abderrahmen

Add features:
- Number of word in e-mail
- ...
"""

# Import
from sklearn.naive_bayes import GaussianNB
import numpy
from numpy import genfromtxt

# Import spam data
##################

# Additional features
an = 1

# Load word stats
word_stats = genfromtxt('data\Spam\emails-train-features.txt', delimiter=' ')
word_stats_a = numpy.array(word_stats)
nb_doc = int(max(word_stats_a[:,0]))
doc_stats = range(0,nb_doc)

# Load word stats test
word_stats_test = genfromtxt('data\Spam\emails-test-features.txt', delimiter=' ')
word_stats_a_test = numpy.array(word_stats_test)
nb_doc_test = int(max(word_stats_a_test[:,0]))
doc_stats_test = range(0,nb_doc_test)

# Load label stats
labels = genfromtxt('data\Spam\emails-train-labels.txt', delimiter=' ')
labels_a = numpy.array(labels)
print('labels:')
print(labels)

# Load label stats test
labels_test = genfromtxt('data\Spam\emails-test-labels.txt', delimiter=' ')
labels_a_test = numpy.array(labels_test)
print('labels_test:')
print(labels_test)

# Count number of words in the dictionary
word_list = word_stats_a[:,1]
nb_words = int(max(word_list))

print('nb_words:')
print(nb_words)

# Build an input for the machine
for i in range(0,nb_doc):
    doc_stats[i] = [0]*(nb_words+an)

for stat in word_stats_a:
    doc_id = int(stat[0]-1)
    word_id = int(stat[1]-1)
    nb_word = int(stat[2]-1)
    doc_stats[doc_id][word_id] = nb_word
    doc_stats[doc_id][nb_words+1-1] += nb_word # Nb words in e-mail
    
    
# Build an input for the machine test
for i in range(0,nb_doc_test):
    doc_stats_test[i] = [0]*(nb_words+an)

for stat in word_stats_a_test:
    doc_id = int(stat[0]-1)
    word_id = int(stat[1]-1)
    nb_word = int(stat[2]-1)
    doc_stats_test[doc_id][word_id] = nb_word
    doc_stats_test[doc_id][nb_words+1-1] += nb_word # Nb words in e-mail


# Generate model
gnb = GaussianNB()
model = gnb.fit(doc_stats, labels_a)

# Predict label (spam/no spam)
y_pred = model.predict(doc_stats_test)

# Print model results
print('y_pred:')
print(y_pred)

# Print score
print('Score:')
s = model.score(doc_stats_test,labels_test)
print(s)

#print('iris.target:')
#print(iris.target)

#print('y_pred:')
#print(y_pred)
