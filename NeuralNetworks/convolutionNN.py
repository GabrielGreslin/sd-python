from math import sqrt
from pybrain import LinearLayer, FullConnection, SigmoidLayer, BiasUnit, MotherConnection, SharedFullConnection, \
    LinearConnection
from pybrain.structure.moduleslice import ModuleSlice

__author__ = 'Gabriel'

'''
Create automatically piece of the neural network with pybrain
'''

'''
Assert the square(inputNode) is a multiple ofsizewindow
Connect with a bias node
'''
def createConvolutionnalLayer(neuralnet, inputlayer, sizewindow):

    convolutionWindowsSize = sizewindow**2
    square_side_input_number = int(sqrt(inputlayer.outdim))
    nb_of_conv_axis = square_side_input_number - sizewindow + 1
    nb_of_conv_tot = nb_of_conv_axis**2

    convlayer = SigmoidLayer(nb_of_conv_tot, name="convLayer")
    biasLayer = BiasUnit(name="Bias")
    neuralnet.addModule(convlayer)
    neuralnet.addModule(biasLayer)

    for i in range(0,nb_of_conv_axis): #for each conv row
        for j in range(0,nb_of_conv_axis): #for each conv column
            inputSlice = []
            for k in range(0,sizewindow): #for each input column
                begin = (i+k)*square_side_input_number+j
                end = (i+k)*square_side_input_number+j+sizewindow
                inputNeurons = ModuleSlice(inputlayer,outSliceFrom=begin,outSliceTo=end)
                inputSlice.append(inputNeurons)

            destination = ModuleSlice(convlayer,inSliceFrom=i*nb_of_conv_axis+j,inSliceTo=i*nb_of_conv_axis+j+1)
            neuralnet.addConnection(FullConnection(biasLayer, destination, name=str(i)+"_"+str(j)+"_Bias"))

            for k,inSl in enumerate(inputSlice):
                inToconvLayer = FullConnection(inSl, destination, name=str(i)+"_"+str(j)+"_"+str(k))
                neuralnet.addConnection(inToconvLayer)

    print(convlayer.indim)
    print(convlayer.outdim)
    print(convlayer.paramdim)

    return convlayer

'''
Assert the square(inputNode) is a multiple ofsizewindow
Connect with a bias node
share the same weights
'''
def createConvolutionnalSharedWeightLayer(neuralnet, inputlayer, sizewindow):

    convolutionWindowsSize = sizewindow**2
    square_side_input_number = int(sqrt(inputlayer.outdim))
    nb_of_conv_axis = square_side_input_number - sizewindow + 1
    nb_of_conv_tot = nb_of_conv_axis**2

    convlayer = SigmoidLayer(nb_of_conv_tot, name="convLayer")
    biasLayer = BiasUnit(name="Bias")
    neuralnet.addModule(convlayer)
    neuralnet.addModule(biasLayer)

    #Create a mother per line
    mothers = []
    #+1 biais
    mothers.append(MotherConnection(1,name='mother_bias'))
    for l in range(0,sizewindow):
        mothers.append(MotherConnection(sizewindow,name='mother_'+str(l)))

    for i in range(0,nb_of_conv_axis): #for each conv row
        for j in range(0,nb_of_conv_axis): #for each conv column
            inputSlice = []
            for k in range(0,sizewindow): #for each input column
                begin = (i+k)*square_side_input_number+j
                end = (i+k)*square_side_input_number+j+sizewindow
                inputNeurons = ModuleSlice(inputlayer,outSliceFrom=begin,outSliceTo=end)
                inputSlice.append(inputNeurons)

            neuronIndice = i*nb_of_conv_axis+j
            destination = ModuleSlice(convlayer,inSliceFrom=neuronIndice,inSliceTo=neuronIndice+1,outSliceFrom=neuronIndice,outSliceTo=neuronIndice+1)
            neuralnet.addConnection(SharedFullConnection(mothers[0],biasLayer, destination, name=str(i)+"_"+str(j)+"_Bias"))

            for k,inSl in enumerate(inputSlice):
                inToconvLayer = SharedFullConnection(mothers[k+1],inSl, destination, name=str(i)+"_"+str(j)+"_"+str(k))
                neuralnet.addConnection(inToconvLayer)

    print("Convolution layer")
    print(convlayer.indim)
    print(convlayer.outdim)

    return convlayer

'''
Create Subsampling layer
'''

def createSubSamplingLayer(neuralnet, inputlayer, sizewindow):

    convolutionWindowsSize = sizewindow**2
    square_side_input_number = int(sqrt(inputlayer.outdim))
    nb_of_sub_axis = int(square_side_input_number / sizewindow) # no overlapping
    nb_of_conv_tot = nb_of_sub_axis**2

    subsamplayer = SigmoidLayer(nb_of_conv_tot, name="subSampleLayer")
    biasLayer = BiasUnit(name="Bias")
    neuralnet.addModule(subsamplayer)
    neuralnet.addModule(biasLayer)

    for i in range(0,nb_of_sub_axis): #for each subsampling row
        for j in range(0,nb_of_sub_axis): #for each subsampling column
            inputSlice = []
            for k in range(0,sizewindow): #for each input column
                begin = (i+k)*square_side_input_number+j*sizewindow
                end = (i+k)*square_side_input_number+j*sizewindow+sizewindow
                inputNeurons = ModuleSlice(inputlayer,outSliceFrom=begin,outSliceTo=end)
                inputSlice.append(inputNeurons)

            neuronIndice = i*nb_of_sub_axis+j
            destination = ModuleSlice(subsamplayer,inSliceFrom=neuronIndice,inSliceTo=neuronIndice+1,outSliceFrom=neuronIndice,outSliceTo=neuronIndice+1)
            neuralnet.addConnection( FullConnection(biasLayer, destination, name=str(i)+"_"+str(j)+"_Bias"))

            for k,inSl in enumerate(inputSlice):
                inToconvLayer = FullConnection(inSl, destination, name=str(i)+"_"+str(j)+"_"+str(k))
                neuralnet.addConnection(inToconvLayer)

    print("Sub sampling Layer")
    print(subsamplayer.indim)
    print(subsamplayer.outdim)

    return subsamplayer

'''
Create Subsampling layer with shared weight
'''
def createSubSamplingLayerSharedWeight(neuralnet, inputlayer, sizewindow):

    convolutionWindowsSize = sizewindow**2
    square_side_input_number = int(sqrt(inputlayer.outdim))
    nb_of_sub_axis = int(square_side_input_number / sizewindow) # no overlapping
    nb_of_conv_tot = nb_of_sub_axis**2

    subsamplayer = SigmoidLayer(nb_of_conv_tot, name="subSampleLayer")
    biasLayer = BiasUnit(name="Bias")
    neuralnet.addModule(subsamplayer)
    neuralnet.addModule(biasLayer)

    #Create a mother per line
    mothers = []
    #+1 biais
    mothers.append(MotherConnection(1,name='mother_bias'))
    for l in range(0,sizewindow):
        mothers.append(MotherConnection(sizewindow,name='mother_'+str(l)))

    for i in range(0,nb_of_sub_axis): #for each subsampling row
        for j in range(0,nb_of_sub_axis): #for each subsampling column
            inputSlice = []
            for k in range(0,sizewindow): #for each input column
                begin = (i+k)*square_side_input_number+j*sizewindow
                end = (i+k)*square_side_input_number+j*sizewindow+sizewindow
                inputNeurons = ModuleSlice(inputlayer,outSliceFrom=begin,outSliceTo=end)
                inputSlice.append(inputNeurons)

            neuronIndice = i*nb_of_sub_axis+j
            destination = ModuleSlice(subsamplayer,inSliceFrom=neuronIndice,inSliceTo=neuronIndice+1,outSliceFrom=neuronIndice,outSliceTo=neuronIndice+1)
            neuralnet.addConnection( SharedFullConnection(mothers[0],biasLayer, destination, name=str(i)+"_"+str(j)+"_Bias"))

            for k,inSl in enumerate(inputSlice):
                inToconvLayer = SharedFullConnection(mothers[k+1],inSl, destination, name=str(i)+"_"+str(j)+"_"+str(k))
                neuralnet.addConnection(inToconvLayer)

    print("Sub sampling Layer Shared weight")
    print(subsamplayer.indim)
    print(subsamplayer.outdim)

    return subsamplayer
