import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(x))

class Neuron:

    def __init__(self, weights = [0, 1], bias = 0):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        in1, in2 = inputs
        wh1, wh2 = self.weights
        return sigmoid(in1 * wh1 + in2 * wh2 + self.bias)

class NeuralNetwork:

    def __init__(self, inputLayer, hiddenLayer, outputN):
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputN = outputN

    def process(self, inputs):
        in1, in2 = inputs
        inN1, inN2 = self.inputLayer
        out1 = inN1.feedForward([in1, in2])
        out2 = inN2.feedForward([in1, in2])

        hN1, hN2 = self.hiddenLayer
        inter1 = hN1.feedForward([out1, out2])
        inter2 = hN2.feedForward([out1, out2])

        outputN = self.outputN

        return outputN.feedForward([inter1, inter2])

inN1 = Neuron()
inN2 = Neuron()
hN1 = Neuron()
hN2 = Neuron()
outputN = Neuron()

NN = NeuralNetwork([inN1, inN2], [hN1, hN2], outputN)
print(NN.process([2,3]))