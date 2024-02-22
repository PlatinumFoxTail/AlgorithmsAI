import unittest
from unittest.mock import patch, MagicMock
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

    #with patch decorator calls will be replaced with mock object
    @patch('network.random.shuffle')
    @patch('builtins.print')
    def test_SGD_with_test_data(self, mock_print, mock_shuffle):
        #mocking shuffle to avoid actual shuffling
        mock_shuffle.side_effect = lambda x: x
        
        #mocking print to avoid actual printing
        mock_print.side_effect = MagicMock()

        #list of tuples, where each tuple consist of two numpy arrays. 1st array represents input data and 2nd array desired output data. Process repeated x10 for training_data and x5 for test data.
        training_data = [(np.random.randn(self.sizes[0], 1), np.random.randn(self.sizes[-1], 1)) for _ in range(10)]
        test_data = [(np.random.randn(self.sizes[0], 1), np.random.randn(self.sizes[-1], 1)) for _ in range(5)]

        #numpy arrays converted to scalar values for test_data
        test_data_scalar = [(x.flatten(), y.flatten()) for (x, y) in test_data]

        #input data reshaped to ensure it's a column vector
        test_data_reshaped = [(x.reshape(-1, 1), y.reshape(-1, 1)) for (x, y) in test_data_scalar]

        #calling SGD methdo from Network class
        self.network.SGD(training_data, epochs=5, mini_batch_size=2, eta=0.1, test_data=test_data_reshaped)
        
        #checking print was called for all 5 epochs
        self.assertEqual(mock_print.call_count, 5)
    
    '''#with patch decorator calls will be replaced with mock object
    @patch('network.random.shuffle')
    @patch('builtins.print')
    def test_SGD_without_test_data(self, mock_print, mock_shuffle):
        
        mock_shuffle.side_effect = lambda x: x  
        mock_print.side_effect = MagicMock()

        training_data = [(np.random.randn(self.sizes[0], 1), np.random.randn(self.sizes[-1], 1)) for _ in range(10)]

        self.network.SGD(training_data, epochs=5, mini_batch_size=2, eta=0.1)
        
        self.assertEqual(mock_print.call_count, 5)'''