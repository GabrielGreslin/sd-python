__author__ = 'Gabriel'

from numpy import genfromtxt
import numpy as np

#dataColumnNum = (3+457+495+472+111+19)
dataColumnNum = 1558

dtypeAd = "f4,"* dataColumnNum
dtypeAd = dtypeAd + "S5"
filling_values = [0 for x in range(0,dataColumnNum+2)]
filling_values[0]=-1
filling_values[1]=-1
filling_values[2]=-1
filling_values[3]=-1

data = genfromtxt("""..\data\internetAds\\ad.data""", delimiter=',', dtype=dtypeAd,skip_header=0,missing_values ="?",filling_values=filling_values)

nbInstance = len(data)
print(nbInstance)

def fromAdsDataToXY(data):

    X = np.zeros((nbInstance,dataColumnNum),dtype=float)
    Y = np.zeros((nbInstance,1),dtype=bool)

    summation = [0,0,0]
    counter = [0,0,0]
    avr = [0,0,0]

    for i,l in enumerate(data):
        for j in range(0,3):
            if(l[j] != -1 ):
                summation[j]+=l[j]
                counter[j]+=1

        print(l)
        print(len(l))
        print(X.shape)
        X[i,:] = (l[:-1])
        Y[i,0] = (l[-1] == "ad.")


    for i in range(0,3):
        avr = summation[i]/counter[i]



    return (X,Y)

(X,Y)=fromAdsDataToXY(data)
print(X)
print(Y)