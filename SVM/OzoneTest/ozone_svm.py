from datetime import date
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

__author__ = 'Gabriel'

from numpy import genfromtxt
import numpy as np

data = genfromtxt('..\data\Ozone\ozone.dat', delimiter=' ', dtype=None,skip_header=1)

head = ("Jour ferie ou pas","quantiteOzoneAir")

threshold = 150 # sur 03obs

def uniqueStringIdentifier(str):
    if not hasattr(uniqueStringIdentifier,"dictOfExistingString"):
        uniqueStringIdentifier.dictOfExistingString = {}
        uniqueStringIdentifier.counter = 0

    try:
        key = uniqueStringIdentifier.dictOfExistingString[str]
    except (KeyError):
        uniqueStringIdentifier.dictOfExistingString[str] = uniqueStringIdentifier.counter
        key = uniqueStringIdentifier.counter
        uniqueStringIdentifier.counter = uniqueStringIdentifier.counter +1

    return key

def builtX(data):

    columnsUsedArray = [0]+[x for x in range(2,10)];
    columnsUsedTuple = [x for x in columnsUsedArray]

    samples_number = len(data)
    feature_number = 9

    X = np.zeros((samples_number,feature_number),dtype=float)

    for i,x in enumerate(data):
        line = [x[j] for j in columnsUsedTuple]
        line[-3] = uniqueStringIdentifier(line[-3])
        X[i,:] = line

    return X

X = builtX(data)


def builtY(data):
    samples_number = len(data)
    Y = np.zeros(samples_number,dtype=bool)

    for i,x in enumerate(data):
        Y[i] = x[1]>threshold

    return Y

Y = builtY(data)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Generate model
svc = svm.SVC(class_weight="balanced",C=0.45,kernel = 'linear')#{True:"balanced",False:"balanced"})
svc.fit(X_train, y_train)
pred = svc.predict(X_test)

if(y_test[0] == True):
    t_names = ["Ozone Day","Less Poluted"]
else:
    t_names = ["Less Poluted","Ozone Day"]


cr = classification_report(y_test,pred,target_names=t_names)
print(cr)

ac = confusion_matrix(y_test, pred, labels=[True,False])
print(ac)

'''
1st attemps
              precision    recall  f1-score   support

   Ozone Day       0.82      0.99      0.90       211
Less Poluted       0.67      0.08      0.14        50

 avg / total       0.79      0.82      0.75       261

[[  4  46]
 [  2 209]]

 Issues with the number of sample in each class
 50 samples in the ozone day class
 211 in the non ozone day class

'''

'''
2nd attemps with balanced class weight

              precision    recall  f1-score   support

   Ozone Day       0.84      0.95      0.89       211
Less Poluted       0.52      0.24      0.33        50

 avg / total       0.78      0.81      0.78       261

[[ 12  38]
 [ 11 200]]
'''

'''
C = 0.4
              precision    recall  f1-score   support

   Ozone Day       0.86      0.95      0.91       211
Less Poluted       0.64      0.36      0.46        50

 avg / total       0.82      0.84      0.82       261

[[ 18  32]
 [ 10 201]]

 But,at this point, we have a C parameter for fitting that is good for the particular test set (! dangerous)
'''


'''
Kernels test
all above with
sigmoid failed with everything classified as not ozone
linear
              precision    recall  f1-score   support

   Ozone Day       0.96      0.77      0.86       211
Less Poluted       0.48      0.88      0.62        50

 avg / total       0.87      0.79      0.81       261

[[ 44   6]
 [ 48 163]]
 -> better if we dont want to miss an ozone day

'''