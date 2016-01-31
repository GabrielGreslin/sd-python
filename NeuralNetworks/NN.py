__author__ = 'Gabriel'

from numpy import genfromtxt
import numpy as np

#from sklearn.neural_network import MLPClassifier # not implemented in sci learn version 0.17
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.supervised.trainers import BackpropTrainer

import matplotlib.pyplot as plt


data = genfromtxt("..\data\Optical character recognition\optdigits.tra", delimiter=',', dtype=None,skip_header=0)
print(data.shape)
datatest = genfromtxt("..\data\Optical character recognition\optdigits.tes", delimiter=',', dtype=None,skip_header=0)


#Last row is the label (64 features + 1 label)

'''
Improvement :
Layer join on subcells above
Then second hidden layer
Then result
'''

'''
Load the data in a X and Y matrice, normalized
'''
def builtXndY(data):
    samples_number = len(data)
    feature_number = 64
    X = np.zeros((samples_number,feature_number),dtype=float)
    Y = np.zeros((samples_number,1),dtype=float)

    for i,l in enumerate(data):
        X[i,:] = (l[:-1]-8)/16 # normalization
        Y[i,0]= l[-1]

    return (X,Y)

(X,Y)= builtXndY(data)
(Xtest,Ytest)= builtXndY(datatest)

# Generate the neural network model
nn = FeedForwardNetwork()
inLayer = LinearLayer(64, name="Input")
hiddenLayer = SigmoidLayer(50, name="First hidden layer")
outLayer = LinearLayer(10, name="Output")

nn.addInputModule(inLayer)
nn.addModule(hiddenLayer)
nn.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer,name="c1")
hidden_to_out = FullConnection(hiddenLayer, outLayer,name="c2")

nn.addConnection(in_to_hidden)
nn.addConnection(hidden_to_out)

nn.sortModules()

print(nn)
#Activate for a particular value
#print(nn.activate([0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]))

alldata = ClassificationDataSet(64,nb_classes=10, class_labels= [str(x) for x in range(0,10)] )
alldata.setField("input",X)
alldata.setField("target",Y)

alldatatest  = ClassificationDataSet(64,nb_classes=10, class_labels= [str(x) for x in range(0,10)] )
alldatatest.setField("input",Xtest)
alldatatest.setField("target",Ytest)

alldata._convertToOneOfMany( )
alldatatest._convertToOneOfMany( )

print("Number of training patterns: ", len(alldata))
print("Input and output dimensions: ", alldata.indim, alldata.outdim)
print("First sample (input, target, class):")
print(alldata['input'][0], alldata['target'][0], alldata['class'][0])
#normalize data

trainer = BackpropTrainer( nn, dataset=alldata, learningrate=0.01,momentum=0.1, verbose=False, weightdecay=0.002,batchlearning=False)

x_errors = np.zeros((1,20),dtype=int)
errors = np.zeros((2,20),dtype=float)

for i in range(20):
    trainer.trainEpochs( 1 )
    trnresult = percentError( trainer.testOnClassData(),alldata['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=alldatatest ), alldatatest['class'] )

    x_errors[0,i]= trainer.totalepochs
    errors[0,i]= trnresult
    errors[1,i]= tstresult
    print("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult)


plt.plot(x_errors[0,:],errors[0,:],'g',x_errors[0,:],errors[1,:],'b')
plt.show()
