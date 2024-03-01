# Execution documentation

This document contains the overview of the created program.

## General structure of program

The program predicts handwritten digits from the MNIST dataset. The actual digit prediction is done in the user-interface (digit_predicter.ipynb), where the neural network (network.py) and MNIST image loader (mnist_loader.py) are utilized. In the user-interface the user can select what MNIST digit to be predicted by the neural network.

The network is tested in benchmark_networks.py, where same testing conditions is set up for the network and a simple network (simple_network.py) for comparsion. More information on how to run the so called sanity check is described in [Testing documentation](https://github.com/PlatinumFoxTail/MachineLearning_NeuralNetwork/blob/main/documents/testing_documentation.md). 

## Performance

The prediction accuracy for the network can be find in the user-interface e.g. if setting the neural network to 784 input neurons, 30 hidden layer neurons, 10 output neurons, 30 epochs, mini-batch size to 10, and a learning rate 3.0 the prediction accuracy of approx. 94% is achieved. More info regarding prediction accuracy of the neural network in [Testing documentation](https://github.com/PlatinumFoxTail/MachineLearning_NeuralNetwork/blob/main/documents/testing_documentation.md).

In the sanity check it seems that the neurons' outputs from the hidden layer and output layer is following the same pattern when comparing the simple network with the network. More info about the results are described in the [Testing documentation](https://github.com/PlatinumFoxTail/MachineLearning_NeuralNetwork/blob/main/documents/testing_documentation.md) 

## Future improvements

In future it would be interesting to apply the created neural network on other image prediction tasks (e.g. brain imaging) and develop further other types of neural networks more suitable for those specific tasks (e.g. convolutional neural networks).

## References

The key source for generating the program is the following reference:
* [Neural Networks and Deep Learning ](http://neuralnetworksanddeeplearning.com/chap1.html)

Other used sources to support the key source are:

* [Wikipedia, Multilayer perceptron ](https://en.wikipedia.org/wiki/Multilayer_perceptron)
* [Mathemathics for Machine learning ](https://mml-book.github.io/)
