"""Multilayer perceptron class module.

Defines multilayer perceptron neural network model.
"""
import numpy as np
from logger import LOGGER

def logistic(x):
    """Logistic function."""
    return 1/(1+np.exp(-x))

def logistic_d(x):
    """Logistic function derivative."""
    return logistic(x)*(1-logistic(x))

def squared_error(y, d):
    """Return squared error between given vectors multiplied by 0.5."""
    return 0.5*((y-d)**2).sum()


class MLP(object):
    """Multilayer perceptron class.

    Attributes:
        weights (numpy.ndarray): Weights matrix. Bias weight included.
        activation (function): Activation function.
        activation_d (function): Activation function derivative.
    """
    def __init__(self, structure, objective=squared_error, activation=logistic,
                 activation_d=logistic_d, learn_coef=0.2):
        """Initializes MLP with random weights.

        Args:
            structure (list): List containing number of nodes (e.g. [2, 2, 1]
                                stands for 2 inputs, 2 neurons in hidden layer,
                                1 output).
            activation (function): Activation function.
            activation_d (function): Activation function derivative.
        """
        np.random.seed(1)
        self.weights = [np.random.rand(structure[i-1], structure[i])
                        for i in range(1, len(structure))]
        self.y = [np.ndarray(output) for output in structure]
        self.z = [np.ndarray(output) for output in structure]
        self.activation = activation
        self.activation_d = activation_d
        self.objective = objective
        self.learn_coef = learn_coef

    def predict_single(self, x_in):
        """Predicts output vector based on a given single input vector.
        Adds bias value to the beginning of input vector.

        Args:
            x_in (numpy.ndarray): Input vector (without bias).

        Returns:
            numpy.ndarray: Predicted output vector.
        """
        self.y[0] = self.z[0] = x_in
        for i in range(len(self.weights)):
            self.z[i+1] = np.dot(self.y[i], self.weights[i])
            self.y[i+1] = self.activation(self.z[i+1])
        return self.y[-1]

    def train_single(self, x_in, d_out):
        """Predicts output for a given input vector (single sample).
        Calculates error between predicted vector and correct given d_out
        vector.
        Corrects weights using backpropagation method.

        Args:
            x_in (numpy.ndarray): Input vector (without bias)
            d_out (numpy.ndarray): Output vector (correct answer for x_in)
        """
        self.predict_single(x_in)
        delta = -(self.y[-1]-d_out)*self.activation_d(self.z[-1])
        for i in range(-1, -len(self.weights)-1, -1):
            djdw = np.dot(self.y[i-1][np.newaxis].T, delta[np.newaxis])
            delta = np.dot(delta, self.weights[i].T)*self.activation_d(self.z[i-1])
            self.weights[i] = self.weights[i] + self.learn_coef*djdw

if __name__ == '__main__':
    inputs = np.array(([1, 1], [1, 0], [0, 1], [0, 0]), dtype=float)
    outputs = np.array(([0], [1], [1], [0]), dtype=float)
    net = MLP([2, 2, 1])
    print("Should get: ", outputs)
    for i in range(10000):
        for i, o in zip(inputs, outputs):
            net.train_single(i, o)
    result = net.predict_single(inputs[0])
    print("Iteration: ", i, " Got: ", result)
    result = net.predict_single(inputs[1])
    print("Iteration: ", i, " Got: ", result)
    result = net.predict_single(inputs[2])
    print("Iteration: ", i, " Got: ", result)
    result = net.predict_single(inputs[3])
    print("Iteration: ", i, " Got: ", result)
