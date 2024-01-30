import unittest
from network import Network

class TestNetwork(unittest.TestCase):
    def setUp(self):
        print("Set up goes here")

    def test_constructor_setting_network_right(self):
        sizes = [5, 4, 3]
        network = Network(sizes)
        
        self.assertEqual(network.num_layers, len(sizes))
        self.assertEqual(network.sizes, sizes)

