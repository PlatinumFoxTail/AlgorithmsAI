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
                        for x, y in zip(sizes[:-1], sizes[1:])]
        '''
    
    #feedforward method, where output from one layer is used as input to the next layer
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            #np.dot(w, a) is dot product of arrays w, and a
            a = sigmoid(np.dot(w, a) + b)
        return a
        
    #stochastic gradient descent (SGD) method, where Network object is learning
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        #list of tuples with training input and desired output
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        #epochs is number of complete passes through training_data
        for j in range(epochs):
            random.shuffle(training_data)
            #mini-batch is subset of complete training_data
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                #eta is learning rate
                self.update_mini_batch(mini_batch, eta)
            #tracking partial partial progress
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    #applying single step of gradient descent using backpropagation to update network's weights and biases
    def update_mini_batch(self, mini_batch, eta):
        #initializing two arrays of zeros to store the gradients of the cost function for biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #backpropagation algorithm
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
    #calculating (nabla_b, nabla_w) tuple, which are layer-by-layer lists of numpy arrays and representing gradient of cost function C_x
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        #feedforward pass, calculating activations and wighted inputs for each layer
        activation = x
        #initiating storing all activations layer-by-layer
        activations = [x]
        #initiating storing all z vectors, layer-by-layer
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward pass, calculating error delta at output layer and propagates it backward to compute gradients of the cost function with respect to biases and weights
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l=1: last layer of neurons, l=2: second last layer of neurons, and so forth.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)
    
    #returns vector of partial derivatives of the cost with respect to the output activations
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    #calcualting correct output/test inputs
    def evaluate(self, test_data):
        #neural network's output is index of neuron in final layer with highest activation
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)