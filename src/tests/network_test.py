import unittest
import numpy as np
from network import Network

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.sizes = [5, 4, 3]
        self.network = Network(self.sizes)
        self.input_data = np.random.randn(self.sizes[0], 1)
        self.eta = 0.1

    def test_constructor_setting_network_right(self):
        
        self.assertEqual(self.network.num_layers, len(self.sizes))
        self.assertEqual(self.network.sizes, self.sizes)
    
    def test_feedforward_output_shape_and_value(self):
        output = self.network.feedforward(self.input_data)

        self.assertEqual(output.shape, (self.sizes[-1], 1))
        self.assertTrue(all(0 <= value <= 1 for value in output))

    def test_update_mini_batch_weights_and_biases_revised(self):
        #creating a mini-batch with a single training random example
        mini_batch = [(np.random.randn(self.sizes[0], 1), np.random.randn(self.sizes[-1], 1))]
        
        #saving the  starting weights and biases
        starting_weights = [np.copy(w) for w in self.network.weights]
        starting_biases = [np.copy(b) for b in self.network.biases]
        
        #revising weights and biases
        self.network.update_mini_batch(mini_batch, self.eta)
        
        #checking weights and biases updated
        for i in range(len(self.network.weights)):
            self.assertFalse(np.array_equal(starting_weights[i], self.network.weights[i]))
        for i in range(len(self.network.biases)):
            self.assertFalse(np.array_equal(starting_biases[i], self.network.biases[i]))

    # CONTINUE TO MAKE UNIT TEST FOR SGD.
    
    #Test working, but not actually improving the coverage rate of network.py
    '''def test_initializing_biases_and_weights(self):
        
        for layer in range(1, len(self.sizes)):
            self.assertEqual(self.network.biases[layer-1].shape, (self.sizes[layer], 1))

        for layer in range(1, len(self.sizes)):
            self.assertEqual(self.network.weights[layer-1].shape, (self.sizes[layer], self.sizes[layer-1]))'''