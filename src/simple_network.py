import numpy as np

# sigmoid function
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

#derivative of sigmoid function
def sigmoid_prime(z):
    return z * ( 1 - z)

class SimpleNetwork:
    def __init__(self, input, output):
        self.input = input
        self.output = output
        #producing same set of random numbers to ensure reproducibility 
        #for evaluation purposes in benchmark_networks.py
        np.random.seed(1)
        #initializing random weights from -1.0 to 1.0 for a network
        #of 3 neurons in input layer, 4 neurons in hidden layer...
        self.weight_inputhidden = 2 * np.random.random((3, 4)) - 1
        #...and 1 neuron in final layer
        self.weight_hiddenoutput = 2 * np.random.random((4, 1)) - 1

    #training simple_network with learning rate eta
    def train(self, eta):
        hidden_layer_outputs = []
        output_layer_outputs = []

        #30 000 iterations training
        for _ in range(30000):
            input_inputlayer = self.input
            #sigmoid activation function of the matrix multiplication of weights and inputs
            activations_hiddenlayer = sigmoid(np.dot(input_inputlayer, self.weight_inputhidden))
            activations_outputlayer = sigmoid(np.dot(activations_hiddenlayer, self.weight_hiddenoutput))

            #only selecting value corresponding to hidden layer's four neurons.
            #flatten converts to 1-D array and tolist converts to Python list for plotting purpouses in benchmark_networks.py
            hidden_layer_outputs.append(activations_hiddenlayer[0].flatten().tolist())
            #only selecting value corresponding to output layer's only neuron
            output_layer_outputs.append(activations_outputlayer[-1].flatten().tolist())

            output_error = self.output - activations_outputlayer
            #backpropagation of hidden-output weights
            output_delta = output_error * sigmoid_prime(activations_outputlayer)
            
            #backpropagation of error to hidden layer
            hidden_error = output_delta.dot(self.weight_hiddenoutput.T)
            #backpropagation of input-hidden weights
            hidden_delta = hidden_error * sigmoid_prime(activations_hiddenlayer)
            
            #updating weights based on deltas and learning rate eta
            self.weight_hiddenoutput += eta * activations_hiddenlayer.T.dot(output_delta)
            self.weight_inputhidden += eta * input_inputlayer.T.dot(hidden_delta)

        return hidden_layer_outputs, output_layer_outputs