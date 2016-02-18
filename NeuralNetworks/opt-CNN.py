

__author__ = 'Gabriel'

'''
Create a neural network then train in for the character recognition
Use pybrain library
Try to work on convolution and connect the layer in a meaningful way
'''
import sys
sys.path.append('../')
import numpy as np
import opticalCharacterManipulation
from features import *
from math import sqrt

#from sklearn.neural_network import MLPClassifier # not implemented in sci learn version 0.17
from pybrain.structure import FeedForwardNetwork, GaussianLayer
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure.moduleslice import ModuleSlice


from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError

from pybrain.supervised.trainers import BackpropTrainer

from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader

import matplotlib.pyplot as plt
import time
#---------------------------new import - required
from NeuralNetworks.convolutionNN import createConvolutionnalSharedWeightLayer, createSubSamplingLayerSharedWeight
from NeuralNetworks.convolutionNN import createSubSamplingLayer
#

start = time.time()
#Last row is the label (64 features + 1 label)

############################LOAD DATA
(X,Y,Xtest,Ytest) = opticalCharacterManipulation.loadTrainAndTestFeaturesData(True,True)
#(X,Y,Xtest,Ytest) = opticalCharacterManipulation.loadTrainAndTestFeaturesData(False,True,convolution)


############################CREATE NETWORK


# Generate the neural network model
nn = FeedForwardNetwork()

#Input Layer
feature_size = len(X[0])
print("Input size : "+str(feature_size)+" features")
inputLayer = LinearLayer(feature_size, name="Input")
nn.addInputModule(inputLayer)


hiddenLayer1 = createConvolutionnalSharedWeightLayer(nn,inputLayer,sizewindow = 3)
hiddenLayer2 = createConvolutionnalSharedWeightLayer(nn,inputLayer,sizewindow = 3)
hiddenLayer3 = createConvolutionnalSharedWeightLayer(nn,inputLayer,sizewindow = 3)
hiddenLayer4 = createConvolutionnalSharedWeightLayer(nn,inputLayer,sizewindow = 3)
hiddenLayer5 = createConvolutionnalSharedWeightLayer(nn,inputLayer,sizewindow = 3)
hiddenLayer6 = createConvolutionnalSharedWeightLayer(nn,inputLayer,sizewindow = 3)

subsamplinglayer1 = createSubSamplingLayerSharedWeight(nn,hiddenLayer1,sizewindow = 2)
subsamplinglayer2 = createSubSamplingLayerSharedWeight(nn,hiddenLayer2,sizewindow = 2)
subsamplinglayer3 = createSubSamplingLayerSharedWeight(nn,hiddenLayer3,sizewindow = 2)
subsamplinglayer4 = createSubSamplingLayerSharedWeight(nn,hiddenLayer4,sizewindow = 2)
subsamplinglayer5 = createSubSamplingLayerSharedWeight(nn,hiddenLayer5,sizewindow = 2)
subsamplinglayer6 = createSubSamplingLayerSharedWeight(nn,hiddenLayer6,sizewindow = 2)

secondHiddenLayer = SigmoidLayer(10,name="secondHiddenLayer")
nn.addModule(secondHiddenLayer)

nn.addConnection(FullConnection(subsamplinglayer1, secondHiddenLayer, name="conv1 to hidden"))
nn.addConnection(FullConnection(subsamplinglayer2, secondHiddenLayer, name="conv2 to hidden"))
nn.addConnection(FullConnection(subsamplinglayer3, secondHiddenLayer, name="conv3 to hidden"))
nn.addConnection(FullConnection(subsamplinglayer4, secondHiddenLayer, name="conv4 to hidden"))
nn.addConnection(FullConnection(subsamplinglayer5, secondHiddenLayer, name="conv5 to hidden"))
nn.addConnection(FullConnection(subsamplinglayer6, secondHiddenLayer, name="conv6 to hidden"))


#OutputLayer
classnumber = 10
outputLayer = SigmoidLayer(classnumber, name="Ouput")
nn.addOutputModule(outputLayer)

nn.addConnection(FullConnection(secondHiddenLayer, outputLayer, name="hidden2 to out"))

'''
#-----------------------------------------------
#Create first layers
feature_size = len(X[0])
convolutionWindowsSize = 2*2 +1 #+1 for the bias

inputsLayers = []
nb_of_conv = int((feature_size) / convolutionWindowsSize)
print("First layer convolution : neuron number " + str(nb_of_conv))

inputLayer = LinearLayer(feature_size, name="Input")
nn.addInputModule(inputLayer)

for i in range(0,nb_of_conv):
    inputLay = ModuleSlice(inputLayer,outSliceFrom=i*convolutionWindowsSize,outSliceTo=(i+1)*convolutionWindowsSize)
    inputsLayers.append(inputLay)

for i in range(0,nb_of_conv):
    l = LinearLayer(convolutionWindowsSize, name="Input_" + str(i))
    inputsLayers.append(l)
    nn.addInputModule(l)




############OutputLayer Creation
outLayer = LinearLayer(10, name="Output")
nn.addOutputModule(outLayer)

############Hidden Layer Creation
side_nb_conv = int(sqrt(nb_of_conv))
conv_size_secondlayer =2
nbOfHiddenNeuron = (side_nb_conv-conv_size_secondlayer)**2

print("Hidden layer convolution : neuron number " + str())
hiddenLayer = SigmoidLayer(1, name="Hidden Layer")


for i in range(0,side_nb_conv-conv_size_secondlayer):
    for j in range(0,side_nb_conv-conv_size_secondlayer):

        #create new layer
        name = "H"+str(i)+"_"+str(j)
        nn.addModule(hiddenLayer)
        ModuleSlice(hiddenLayer,)
        hidden_to_out = FullConnection(hiddenLayer, outLayer,name=name+"_out")
        nn.addConnection(hidden_to_out)

        #Connect input Layer to the hidden layer
        for k in range(0,conv_size_secondlayer):
            for l in range(0,conv_size_secondlayer):
                inLay = inputsLayers[(i+k)*side_nb_conv + (j+l)]
                c = FullConnection(inLay,hiddenLayer,name="c"+str((i+k)*side_nb_conv + (j+l))+"_"+str(i)+","+str(j))
                nn.addConnection(c)

'''
nn.sortModules()

print(nn)

alldata = ClassificationDataSet(feature_size,nb_classes=10, class_labels= [str(x) for x in range(0,10)] )
alldata.setField("input",X)
alldata.setField("target",Y)

alldatatest  = ClassificationDataSet(feature_size,nb_classes=10, class_labels= [str(x) for x in range(0,10)] )
alldatatest.setField("input",Xtest)
alldatatest.setField("target",Ytest)

alldata._convertToOneOfMany( )
alldatatest._convertToOneOfMany( )

print("Number of training patterns: ", len(alldata))
print("Input and output dimensions: ", alldata.indim, alldata.outdim)
print("First sample (input, target, class):")
print(alldata['input'][0], alldata['target'][0], alldata['class'][0])

#nn = NetworkReader.readFrom('filename.xml')
trainer = BackpropTrainer( nn, dataset=alldata, learningrate=0.01,momentum=0.1, verbose=False, weightdecay=0.002,batchlearning=False)

numberOfEpoch = 35

x_errors = np.zeros((1,numberOfEpoch),dtype=int)
errors = np.zeros((2,numberOfEpoch),dtype=float)

for i in range(numberOfEpoch):
    trainer.trainEpochs( 1 )
    trnresult = percentError( trainer.testOnClassData(),alldata['class'] )
    tstresult = percentError( trainer.testOnClassData(dataset=alldatatest ), alldatatest['class'] )

    x_errors[0,i]= trainer.totalepochs
    errors[0,i]= trnresult
    errors[1,i]= tstresult
    print("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult)


#plt.plot(x_errors[0,:],errors[0,:],'g',x_errors[0,:],errors[1,:],'b')
#plt.show()

#epoch:   20   train error:  4.34%   test error:  6.84%
#Time elapsed :84.42

#Without normalization of the input values
#epoch:   20   train error:  6.70%   test error: 10.46%
#Time elapsed :80.95
end = time.time()
print("Time elapsed :"+str(end-start))

NetworkWriter.writeToFile(nn, 'base.xml')
