import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derived_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Class Neuron Definition

class Neuron:

    def __init__(self, weights = [1, 1], bias = 0):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        in1, in2 = inputs
        wh1, wh2 = self.weights
        return sigmoid(in1 * wh1 + in2 * wh2 + self.bias)

# Class Network Definition

class NeuralNetwork:

    def __init__(self):

        # self.h1 = Neuron([np.random.normal(), np.random.normal()], np.random.normal())
        # self.h2 = Neuron([np.random.normal(), np.random.normal()], np.random.normal())
        # self.out = Neuron([np.random.normal(), np.random.normal()], np.random.normal())
        self.h1 = Neuron()
        self.h2 = Neuron()
        self.out = Neuron()

    def feedForward(self, inputs):
        out1 = self.h1.feedForward(inputs)
        out2 = self.h2.feedForward(inputs)

        return self.out.feedForward([out1,out2]), out1, out2

data = [['Alice', 'Bob', 'Charlie', 'Diana'],
        [135, 160, 152, 120],
        [65, 72, 70, 60],
        ['F', 'M', 'M', 'F']
        ]

## Data preprocessing

def Preprocess(data):
    for i in range(len(data[1])):
        data[1][i] -= 135

    for i in range(len(data[2])):
        data[2][i] -= 66

    for i in range(len(data[3])):
        if data[3][i] == 'F':
            data[3][i] = 1
        else:
            data[3][i] = 0

Preprocess(data)

## Define Loss function

def MeanSquaredError(pred, true):
    sum = 0
    for i in range(len(pred)):
        sum += (true[i] - pred[i]) ** 2
    return sum / len(pred)

NN = NeuralNetwork()

print(data)

# for i in range(len(data)):
for i in range(1,2):
    o, out1, out2 = NN.feedForward([data[1][i], data[2][i]])
    y_pred = np.round(o)
    MSE = (data[3][i] - y_pred) ** 2

    #L_w1 = L_yPred * yPred_h1 * h1_w1

    #L_yPred = -2(1-yPred)
    L_yPred = -2 * (1 - y_pred)

    #yPred_h1 = w5 * f'(w5*out1 + w6*out2 + bias3)
    w5 = NN.out.weights[0]
    w6 = NN.out.weights[1]
    b3 = NN.out.bias

    yPred_h1 = w5 * derived_sigmoid(w5 * out1 + w6 * out2 + b3)

    #h1_w1 = x1 * f'(w1*x1 + w2*x2 + b1)
    h1_w1 = data[1][i] * derived_sigmoid(data[1][i] * 1 + data[2][i] * 1)

    L_w1 = L_yPred * yPred_h1 * h1_w1

    print(L_w1)











    




