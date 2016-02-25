import sys
import numpy as np
from pybrain.structure.moduleslice import ModuleSlice
import opticalCharacterManipulation

# from sklearn.neural_network import MLPClassifier # not implemented in sci learn version 0.17
from pybrain.structure import FeedForwardNetwork, BiasUnit
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError

from pybrain.supervised.trainers import BackpropTrainer

from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader

import matplotlib.pyplot as plt
import time
# ---------------------------new import - required
from NeuralNetworks.convolutionNN import createConvolutionnalSharedWeightLayer, createSubSamplingLayerSharedWeight, \
    createConvolutionnalLayer, createSubSamplingLayer
sys.path.append('../')
__author__ = 'Gabriel'

'''
Create a neural network then train in for the character recognition
Use pybrain library
Try to work on convolution and connect the layer in a meaningful way
'''

start = time.time()
# Last row is the label (64 features + 1 label)

############################LOAD DATA
(X, Y, Xtest, Ytest) = opticalCharacterManipulation.loadTrainAndTestFeaturesData(True, True)


############################CREATE NETWORK


# Generate the neural network model
nn = FeedForwardNetwork()

# Input Layer
feature_size = len(X[0])
print("Input size : " + str(feature_size) + " features")
inputLayer = LinearLayer(feature_size, name="Input")
nn.addInputModule(inputLayer)

convLayer = createConvolutionnalLayer(nn, inputLayer, sizewindow=2)
convLayer2 = createConvolutionnalLayer(nn, inputLayer, sizewindow=3)


#Add biai
#biaiInput = BiasUnit(name="Biai Input")
#nn.addModule(biaiInput)

#Second hidden layer
secondHiddenLayer = SigmoidLayer(100, name="secondHiddenLayer")
nn.addModule(secondHiddenLayer)


#topSecond = ModuleSlice(secondHiddenLayer,inSliceFrom=0,inSliceTo=50)
#bottomSecond = ModuleSlice(secondHiddenLayer,inSliceFrom=50,inSliceTo=100)
nn.addConnection(FullConnection(inputLayer, secondHiddenLayer, name="input to hidden"))
nn.addConnection(FullConnection(convLayer, secondHiddenLayer, name="conv1 to hidden"))
nn.addConnection(FullConnection(convLayer2, secondHiddenLayer, name="conv1 to hidden"))
#nn.addConnection(FullConnection(biaiInput, secondHiddenLayer, name="biai to hidden"))


#thirdHiddenLayer = SigmoidLayer(100, name="ThirdHiddenLayer")
#nn.addModule(thirdHiddenLayer)

#nn.addConnection(FullConnection(secondHiddenLayer,thirdHiddenLayer,name="Second to third"))
#Add biai
#biaiToOuput = BiasUnit(name="biaiToOuput")
#nn.addModule(biaiToOuput)

# OutputLayer
classnumber = 10
outputLayer = LinearLayer(classnumber, name="Ouput")
nn.addOutputModule(outputLayer)

#nn.addConnection(FullConnection(thirdHiddenLayer, outputLayer, name="hidden2 to out"))
nn.addConnection(FullConnection(secondHiddenLayer, outputLayer, name="hidden2 to out"))
#nn.addConnection(FullConnection(biaiToOuput, outputLayer, name="hidden2 to out"))

nn.sortModules()

print(nn)

alldata = ClassificationDataSet(feature_size, nb_classes=10, class_labels=[str(x) for x in range(0, 10)])
alldata.setField("input", X)
alldata.setField("target", Y)

alldatatest = ClassificationDataSet(feature_size, nb_classes=10, class_labels=[str(x) for x in range(0, 10)])
alldatatest.setField("input", Xtest)
alldatatest.setField("target", Ytest)

alldata._convertToOneOfMany()
alldatatest._convertToOneOfMany()

print("Number of training patterns: ", len(alldata))
print("Input and output dimensions: ", alldata.indim, alldata.outdim)
print("First sample (input, target, class):")
print(alldata['input'][0], alldata['target'][0], alldata['class'][0])

# nn = NetworkReader.readFrom('filename.xml')
trainer = BackpropTrainer(nn, dataset=alldata, learningrate=0.01, momentum=0.1, verbose=False, weightdecay=0.002,
                          batchlearning=False)

numberOfEpoch = 60

x_errors = np.zeros((1, numberOfEpoch), dtype=int)
errors = np.zeros((2, numberOfEpoch), dtype=float)

for i in range(numberOfEpoch):
    trainer.trainEpochs(1)
    trnresult = percentError(trainer.testOnClassData(), alldata['class'])
    tstresult = percentError(trainer.testOnClassData(dataset=alldatatest), alldatatest['class'])

    x_errors[0, i] = trainer.totalepochs
    errors[0, i] = trnresult
    errors[1, i] = tstresult
    current = time.time() - start
    print("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult,\
          " time : %5.1f" % current)

end = time.time()
print("Time elapsed :" + str(end - start))

plt.plot(x_errors[0,:],errors[0,:],'g',x_errors[0,:],errors[1,:],'b')
plt.legend(["train error","test error"])
plt.show()

NetworkWriter.writeToFile(nn, 'base.xml')
