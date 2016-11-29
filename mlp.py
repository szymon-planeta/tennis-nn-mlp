"""Multilayer perceptron class module.

Defines multilayer perceptron neural network model.
"""
import numpy as np
from logger import LOGGER
from statistics import mean

def logistic(x):
    """Logistic function."""
    return 1/(1+np.exp(-x))

def logistic_d(x):
    """Logistic function derivative."""
    return logistic(x)*(1-logistic(x))

def squared_error(y, d):
    """Return squared error between given vectors multiplied by 0.5."""
    return 0.5*((y-d)**2).sum()

def flatten(matrix):
    return np.concatenate([x.ravel() for x in matrix])

def expand(vector, structure):
    matrix = []
    w_end = 0
    for i in range(len(structure)-1):
        w_start = w_end
        size_l = structure[i]
        size_r = structure[i+1]
        w_end = w_end + size_l*size_r
        matrix.append(np.reshape(vector[w_start:w_end],
                                 (size_l, size_r)))
    return matrix


class MLP(object):
    """Multilayer perceptron class.

    Attributes:
        structure (list): List of int - each int is a size of each layer.
        weights ([numpy.ndarray]): List of weights matrixes.
        y ([numpy.ndarray]): List of vectors. Each vector is each layer's
            output (layer 0 output is equal to input vector).
        z ([numpy.ndarray]): List of vectors. Each vector is each layer's
            sum of inputs multiplied by their weights (layer 0 is equal to
            input vector).
        cost_fun (function): Cost function.
        activation_fun (function): Activation function.
        activation_fun_d (function): Activation function derivative.
        learn_coef (float): Learning coefficient.
        mom_coef (float): Momentum coefficient.
    """
    def __init__(self, structure, cost_fun=squared_error, activation_fun=logistic,
                 activation_fun_d=logistic_d, learn_coef=0.2, mom_coef=0.1):
        """Initializes MLP with random weights.

        Args:
            structure (list): List containing number of nodes (e.g. [2, 2, 1]
                                stands for 2 inputs, 2 neurons in hidden layer,
                                1 output).
            cost_fun (function): Cost function.
            activation_fun (function): Activation function.
            activation_fun_d (function): Activation function derivative.
            learn_coef (float): Learning coefficient.
            mom_coef (float): Momentum coefficient.
        """
        self.structure = structure
        self.weights = [np.random.rand(structure[i-1]+1, structure[i])
                        for i in range(1, len(structure))]
        self.y = [np.ndarray(output) for output in structure]
        self.z = [np.ndarray(output) for output in structure]
        self.activation_fun = activation_fun
        self.activation_fun_d = activation_fun_d
        self.cost_fun = cost_fun
        self.learn_coef = learn_coef
        self.mom = [np.zeros(w.shape) for w in self.weights]
        self.mom_coef = mom_coef
        self.correct_predictions = 0

    def predict_single(self, x_in):
        """Predicts output vector based on a given single input vector.
        Adds bias value to the beginning of input vector.

        Args:
            x_in (numpy.ndarray): Input vector (without bias).

        Returns:
            numpy.ndarray: Predicted output vector.
        """
        LOGGER.debug("Got input: %s", x_in)
        self.y[0] = self.add_bias(x_in)
        for i in range(len(self.weights)):
            self.z[i+1] = np.dot(self.y[i], self.weights[i])
            self.y[i+1] = self.activation_fun(self.z[i+1])
            if i < len(self.weights)-1:
                self.y[i+1] = self.add_bias(self.y[i+1])
        LOGGER.debug("Predicted: %s", self.y[-1])
        return self.y[-1]

    def add_bias(self, vector):
        return np.concatenate([[1], vector])

    def get_cost(self, x_in, d_out):
        """Uses predict_single to predict output for given input vector.
        Calculates cost using cost_fun.

        Args:
            x_in (numpy.ndarray): Input vector (without bias).
            d_out (numpy.ndarray): Desired output vector.

        Returns:
            cost (float): Calculated cost.
        """
        self.predict_single(x_in)
        LOGGER.debug("Should get: %s, Got: %s", d_out, self.y[-1])
        result = np.array([1, 0]) if self.y[-1][0] > self.y[-1][1] else np.array([0, 1])
        if np.array_equal(result, d_out):
            self.correct_predictions += 1
        return self.cost_fun(self.y[-1], d_out)

    def train_single(self, x_in, d_out, debug=False):
        """Predicts output for a given input vector (single sample).
        Calculates error between predicted vector and correct given d_out
        vector.
        Backpropagates error to hidden layers.
        Computes gradient and corrects weights.
        If debug mode is enabled, checks if gradient is calculated correctly.

        Args:
            x_in (numpy.ndarray): Input vector (without bias).
            d_out (numpy.ndarray): Desired output vector.
            debug (Boolean): Enables or disable numerical gradient check.
        """
        cost = self.get_cost(x_in, d_out)
        deltas = self.compute_deltas(d_out)
        gradient = self.compute_gradient(deltas)
        if debug:
            num_gradient = self.compute_numerical_gradient(x_in, d_out)
            diff = (flatten(gradient) - num_gradient).sum()**2
            LOGGER.debug("Squared gradient difference: %s", diff)

        self.correct_weights(gradient)

    def start_training(self, X_train, D_train, X_test, D_test, tolerance=0.01,
                      epochs=1000):
        """Online training process of MLP. Uses X_train, D_train as training
        dataset, X_test, D_test as test dataset.

        Args:
            X_train ([numpy.ndarray]): List of numpy.ndarray containing input
                                        vectors for training dataset.
            D_train ([numpy.ndarray]): List of numpy.ndarray containing desired
                                        output for training dataset.
            X_test ([numpy.ndarray]): List of numpy.ndarray containing input
                                        vectors for test dataset.
            D_test ([numpy.ndarray]): List of numpy.ndarray containing desired
                                        output for test dataset.
            tolerance (float): Learning stops if error is lower than this value.
            epochs (float): Learning stops after that many epochs.
        """
        LOGGER.info("Training started.")
        e = tolerance + 1
        epoch = 1
        while e > tolerance and epoch < epochs:
            self.train_multiple(X_train, D_train)
            e = self.test_multiple(X_test, D_test)
            LOGGER.info("Epoch: %s, Current error: %s", epoch, e)
            epoch += 1
        LOGGER.info("Training complete. Error: %s, Epoch: %s", e, epoch)

    def train_multiple(self, X_in, D_out):
        """Invokes self.train_single for every input-output pair given in
        arguments.

        Args:
            X_in ([numpy.ndarray]): List of numpy.ndarray containing input
                                    vectors.
            D_out ([numpy.ndarray]): List of numpy.ndarray containing desired
                                     output vectors.
        """
        for x, d in zip(X_in, D_out):
            self.train_single(x, d)

    def test_multiple(self, X_in, D_out):
        """Predicts output for every vector in X_in, compares predicted output
        with desired output D_out, returns mean of costs.

        Args:
            X_in ([numpy.ndarray]): List of numpy.ndarray containing input
                                    vectors.
            D_out ([numpy.ndarray]): List of numpy.ndarray containing desired
                                     output vectors.

        Returns:
            mean (float): Mean of cost function values.
        """
        self.correct_predictions = 0
        costs = [self.get_cost(x, d) for x, d in zip(X_in, D_out)]
        LOGGER.info("%s%% success on test set.", 100*self.correct_predictions/len(costs))
        return mean(costs)

    def compute_deltas(self, d_out):
        """Backpropagates error. Returns list of errors for each layer except
        hidden layer.

        Args:
            d_out (numpy.ndarray): Desired output vector.

        Returns:
            deltas ([numpy.ndarray]): List of vectors with errors for each
                layer (except input layer).
        """
        deltas = [(self.y[-1]-d_out)*self.activation_fun_d(self.z[-1])]
        for i in range(-1, -len(self.weights), -1):
            deltas.insert(0, np.dot(deltas[i],
                                    self.weights[i].T)[1:]*self.activation_fun_d(self.z[i-1]))
        return deltas

    def compute_gradient(self, deltas):
        """Computes gradient and returns it as a list of matrixes corresponding
        to weights.

        Args:
            deltas ([numpy.ndarray]): List of vectors with errors for each
                layer (except input layer).

        Returns:
            gradient([numpy.ndarray]): List of matrixes corresponding to
                weights.
        """
        gradient = [np.dot(self.y[i][np.newaxis].T, deltas[i][np.newaxis])
                    for i in range(len(self.weights))]
        return gradient

    def get_flat_weights(self):
        """Returns weights as a vector."""
        return flatten(self.weights)

    def set_flat_weights(self, flat_weights):
        """Sets given vector as weights."""
        self.weights = expand(flat_weights, self.structure)

    def compute_numerical_gradient(self, x_in, d_out):
        """Computes gradient numerically. Used for debugging."""
        saved_weights = self.get_flat_weights()
        eps = 1e-4
        gradient = np.zeros(saved_weights.shape)
        disrupt =  np.zeros(saved_weights.shape)

        for i in range(len(disrupt)):
            disrupt[i] = eps

            self.set_flat_weights(saved_weights + disrupt)
            loss1 = self.get_cost(x_in, d_out)

            self.set_flat_weights(saved_weights - disrupt)
            loss2 = self.get_cost(x_in, d_out)

            gradient[i] = (loss1-loss2)/(2*eps)

            disrupt[i] = 0

        self.set_flat_weights(saved_weights)
        return gradient

    def correct_weights(self, gradient, with_momentum=True):
        """Corrects weights values.

        Args:
            gradient ([numpy.ndarray]): Gradient (same shape as self.weights).
            with_momentum (Boolean): If True, saves correction values to
                                     self.mom.
        """
        for i in range(len(self.weights)):
            dweights = -gradient[i]*self.learn_coef + self.mom[i]*self.mom_coef
            self.weights[i] += dweights
            if with_momentum:
                self.mom[i] = dweights


if __name__ == '__main__':
    np.random.seed(1)
    inputs = np.array(([1, 1], [1, 0], [0, 1], [0, 0]), dtype=float)
    outputs = np.array(([0], [1], [1], [0]), dtype=float)
    net = MLP([2, 3, 1], learn_coef=0.2, mom_coef=0.1)
    print("Should get: ", outputs)
#    for i in range(10000):
#        for j, o in zip(inputs, outputs):
#            net.train_single(j, o)
    net.start_training(inputs, outputs, inputs, outputs, epochs=30000,
                       tolerance=0.0001)
    for i in inputs:
        print(net.predict_single(i))
