from sklearn.cross_validation import train_test_split
import sklearn.ensemble as adada
from sklearn.metrics import classification_report, confusion_matrix

__author__ = 'Gabriel'

from numpy import genfromtxt
import numpy as np

#dataColumnNum = (3+457+495+472+111+19)
dataColumnNum = 1559

#Define the data type for each columns
dtypeAd = "f4,"* (dataColumnNum-1)
dtypeAd = dtypeAd + "S5"

#Define the value to fill temporarily the missing values
filling_values = [0 for x in range(0,dataColumnNum+1)]
filling_values[0]=-1
filling_values[1]=-1
filling_values[2]=-1
filling_values[3]=-1

#extract the data
data = genfromtxt("""..\data\internetAds\\ad.data""", delimiter=',', dtype=dtypeAd,skip_header=0,missing_values ="?",filling_values=filling_values)

nbInstance = len(data)
print(nbInstance)

def fromAdsDataToXY(data):

    X = np.zeros((nbInstance,(dataColumnNum-1)),dtype=float)
    Y = np.zeros((nbInstance),dtype=float)

    summation = [0,0,0]
    counter = [0,0,0]
    avr = [0,0,0]

# For each data line
    for i,l in enumerate(data):
        #Computation of the average for the missing columns
        for j in range(0,3):
            if(l[j] != -1 ):
                summation[j]+=l[j]
                counter[j]+=1
        #-------------------------

        #Assignation of each ligne to X and to the end to Y
        X[i,:]= [x for i,x in enumerate(l) if i<(dataColumnNum-1)]
        Y[i] = (l[-1] == b"ad.")
        #-------------------------

    #Average computation and assignation
    for i in range(0,3):
        avr[i] = summation[i]/counter[i]

    for u in range(0,nbInstance):
        for v in range(0,3):
            if X[u,v] == -1:
                X[u,v] = avr[v]
    #-------------------------

    return (X,Y)

(X,Y)=fromAdsDataToXY(data)

#Classifications

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Generate model
classifier = adada.AdaBoostClassifier()#{True:"balanced",False:"balanced"})
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

if(y_test[0] == True):
    t_names = ["Add","Not add"]
else:
    t_names = ["Not add","Add"]


cr = classification_report(y_test,pred,target_names=t_names)
print(cr)

ac = confusion_matrix(y_test, pred, labels=[True,False])
print(ac)
