import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:

    def __init__(self, weights = [0, 1], bias = 0):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        in1, in2 = inputs
        wh1, wh2 = self.weights
        return sigmoid(in1 * wh1 + in2 * wh2 + self.bias)

class NeuralNetwork:

    def __init__(self):
        self.h1 = Neuron()
        self.h2 = Neuron()

        self.out = Neuron()


    def feedForward(self, inputs):  

        out1 = self.h1.feedForward(inputs)
        out2 = self.h2.feedForward(inputs)

        return self.out.feedForward([out1,out2])


NN = NeuralNetwork()
print(NN.feedForward([2,3]))