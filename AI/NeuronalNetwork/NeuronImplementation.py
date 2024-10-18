import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # weight inputs, add bias, then use activation function
        in1, in2 = inputs
        wh1, wh2 = self.weights
        print(in1, in2)
        out = (in1 * wh1) + (in2 * wh2) + self.bias 

        return sigmoid(out)

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x))    # 0.9990889488055994