import random
import numpy as np
from typing import Optional

# sigmoid function
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

#derivative of sigmoid function
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#representation of neural network
class Network(object):

    #intialization of Network object
    def __init__(self, sizes):
        #number of layers in network
        self.num_layers = len(sizes)
        #number of neurons in layer
        self.sizes = sizes

        '''For user interface purpouse i.e. digit_predicter.ipynb'''
        #random intialization to generate Gaussian distributions with mean 0 and standard deviation 1
        #with zip combining x and y into pairs of tuples based on shortest input
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        '''For testing purpouses i.e. benchmark_networks.py
        #producing same set of random numbers to ensure reproducibility 
        #for evaluation purposes in benchmark_networks.py
        np.random.seed(1)
        #setting biases to zeros, because simple_network.py does not have biases
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        #initializing random weights from -1.0 to 1.0 for the weight, as in simple_network.py
        self.weights = [np.random.uniform(-1.0, 1.0, (y, x))
                        for x, y in zip(sizes[:-1], sizes[1:])]'''


    #feedforward method, where output from one layer is used as input to the next layer
    def feedforward(self, input_activations):
        for biases, weights in zip(self.biases, self.weights):
            #np.dot(weights, input_activations) is dot product of arrays weights, and input_activations
            input_activations = sigmoid(np.dot(weights, input_activations) + biases)
        return input_activations

    #stochastic gradient descent (SGD) method, where Network object is learning
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data=None):

        #list of tuples with training input and desired output
        training_data = list(training_data)
        n_training_data = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test_data = len(test_data)

        #epochs is number of complete passes through training_data
        for j in range(epochs):
            random.shuffle(training_data)
            #mini-batch is subset of complete training_data
            mini_batches = [
                training_data[start_index:start_index+mini_batch_size]
                for start_index in range(0, n_training_data, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            #tracking partial partial progress
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test_data))
            else:
                print("Epoch {} complete".format(j))

    def initialize_gradient_arrays(self):
        #initializing two arrays of zeros to store the gradients of the cost function for biases and weights
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        return gradient_b, gradient_w

    #applying single step of gradient descent using backpropagation to update network's weights and biases
    def update_mini_batch(self, mini_batch, learning_rate):
        gradient_b, gradient_w = self.initialize_gradient_arrays()

        for input_activation, expected_output in mini_batch:
            #backpropagation algorithm
            delta_gradient_b, delta_gradient_w = self.backprop(input_activation, expected_output)
            gradient_b = [gradient_b+delta_gradient_b
                          for gradient_b, delta_gradient_b in zip(gradient_b, delta_gradient_b)]
            gradient_w = [gradient_w+delta_gradient_w
                          for gradient_w, delta_gradient_w in zip(gradient_w, delta_gradient_w)]
        self.weights = [weights-(learning_rate/len(mini_batch))*gradient_w
                        for weights, gradient_w in zip(self.weights, gradient_w)]
        self.biases = [biases-(learning_rate/len(mini_batch))*gradient_b
                       for biases, gradient_b in zip(self.biases, gradient_b)]

    #calculating (gradient_b, gradient_w) tuple, which are layer-by-layer lists of...
    #numpy arrays and representing gradient of cost function
    def backprop(self, input_activation, expected_output):
        gradient_b, gradient_w = self.initialize_gradient_arrays()

        #feedforward pass, calculating activations and wighted inputs for each layer
        activation = input_activation
        #initiating storing all activations layer-by-layer
        activations = [input_activation]
        #initiating storing all weighted inputs, layer-by-layer
        weighted_inputs = []
        for biases, weights in zip(self.biases, self.weights):
            weighted_sum = np.dot(weights, activation)+biases
            weighted_inputs.append(weighted_sum)
            activation = sigmoid(weighted_sum)
            activations.append(activation)

        #backward pass, calculating layer error at output layer and propagates it...
        #backward to compute gradients of the cost function with respect to biases...
        #and weights
        layer_error = self.cost_derivative(activations[-1], expected_output) * \
            sigmoid_prime(weighted_inputs[-1])
        gradient_b[-1] = layer_error
        gradient_w[-1] = np.dot(layer_error, activations[-2].transpose())
        #updating the errors and gradients by iterating backward through the layers
        #layer=1: last layer of neurons, layer=2: second last layer of neurons, and so forth.
        for layer in range(2, self.num_layers):
            weighted_sum = weighted_inputs[-layer]
            sp = sigmoid_prime(weighted_sum)
            layer_error = np.dot(self.weights[-layer+1].transpose(), layer_error) * sp
            gradient_b[-layer] = layer_error
            gradient_w[-layer] = np.dot(layer_error, activations[-layer-1].transpose())

        return (gradient_b, gradient_w)

    #returns vector of partial derivatives of the cost with respect to the output activations
    def cost_derivative(self, output_activations, expected_output):
        return output_activations-expected_output

    def evaluate(self, test_data):
        #neural network's output is index of neuron in final layer with highest activation
        test_results = [(np.argmax(self.feedforward(input_data)), label)
                        for (input_data, label) in test_data]
        return sum(int(prediction == label) for (prediction, label) in test_results)