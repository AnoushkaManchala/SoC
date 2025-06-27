inputs = [1.2, 2.3, 3.1]
weights = [0.4, 0.6, -0.9]
bias = 2.0

output = sum([x * w for x, w in zip(inputs, weights)]) + bias
print("Output of the neuron:", output)

def relu(x):
    return max(0, x)

inputs = [-2, -1, 0, 1, 2]
outputs = [relu(i) for i in inputs]
print("ReLU outputs:", outputs)

import math

def softmax(inputs):
    exp_values = [math.exp(i) for i in inputs]
    sum_exp = sum(exp_values)
    return [i / sum_exp for i in exp_values]

inputs = [2.0, 1.0, 0.1]
print("Softmax output:", softmax(inputs))

