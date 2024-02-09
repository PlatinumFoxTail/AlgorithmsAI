# Testing documentation

The testing focus of this project is set on the neural network (network.py), and not other parts such as the user interface (digit_predicter.ipynb). The testing can be divided into two parts: Unit testing & checkstyle and Sanity check. The former is self-explanatory, and the Sanity check includes a comparsion of intermediate outputs from the neural network and a simplified neural network based on simplified MNIST input images.

## Sanity check

### Description

In order to pressure test the network, a simple network was created for comparsion. The simple network (simple_network.py) had 3 neurons in the input layer, 4 neurons in the hidden layer, and 1 neuron in the output layer and no biases. The network, with the same neural network architecture as the simple network, and the simple network were fed with same simple input and output. The input was vectors with 3 elements either 0 or 1 values, and the output values were either 0 or 1. Since output values were chosen as either 0 or 1 values, 1 neuron in the output layer was sufficient. The learing rate was set to 0.01 for both the networks. The weights for both networks were also initialized with same random number.

The outputs from the neurons in the hidden layer and neuron the final layer was recorded during training for both the network and the simple network. The values were plotted for comparsion.

### Results

![Figure 1](https://github.com/PlatinumFoxTail/MachineLearning_NeuralNetwork/blob/main/images/070224.%20network%20vs.%20simple_network.%20eta%200.01%20no%20biases.png)

From the hidden layer outputs in the left-hand side graph in the Figure 1 one can notice that, network's neurons 1&4 increases quite uniformly as a function of iterations while corresponding neurons of the simple network also increases but not as quickly. Further on the simple network's neurons 2&3 decrease quite uniformly as a function of iterations while corresponding neurons of the network also decreases, but not as quickly. Based on this observation it seems that network and the simple network hidden layer outputs correlates to each other, somehow "mirrored" e.g. when network's neuron's 1&4 increase uniformly the simple network's neurons 1&4 increase not as quickly and vice verca when simple network's neurons 2&3 decrease uniformly then network's neurons 2&3 decrease not as quickly. The "mirroring" should not be and issue, when testing the sanity of the network since same pair of neurons increases and decreases for both the netowrk and the simple network.

More importantly it seems in that, both networks output from output layer (right-hand side graph in Figure 1) approaches same value when iterations are sufficient. This gives more confidence that the network should be set up properly.


### Instructions to repeat test

* In the network.py file, the biases and weights need to be uncommented on line 30-39 and commented out on lines 23-28. https://github.com/PlatinumFoxTail/MachineLearning_NeuralNetwork/blob/58606d727707a75be0624a31732c30e3411abfe9/src/network.py#L23-L39
* Run benchmark_networks.py

Initial idea/plan as of 2nd of Feb how to perform sanity check:

* Simplify the MNIST input images e.g. to 10pixels
* Set up a simplified network e.g. ready library
* Train the simplified network and own network with the simplified input
* Compare intermediate calculation steps between the simplified network and own network e.g. feedforward outputs.

## Unit testing and checkstyle

![GHA workflow badge](https://github.com/PlatinumFoxTail/MachineLearning_NeuralNetwork/workflows/CI/badge.svg) 
[![codecov](https://codecov.io/gh/PlatinumFoxTail/MachineLearning_NeuralNetwork/graph/badge.svg?token=4JBGC70B3Z)](https://codecov.io/gh/PlatinumFoxTail/MachineLearning_NeuralNetwork)

For checkstyle pylint has been used. As of 2nd of Feb the code has been rated as 6.70/10.

TO DO:
* Unit test for SGD method, to increase coverage from 77% -> 100%
* Improve pylint score
