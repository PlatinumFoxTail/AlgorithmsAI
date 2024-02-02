# Testing documentation

The testing focus of this project is set on the neural network (network.py), and not other parts such as the user interface (digit_predicter.ipynb). The testing can be divided into two parts: Unit testing & checkstyle and Sanity check. The former is self-explanatory, and the Sanity check includes a comparsion of intermediate outputs from the neural network and a simplified neural network based on simplified MNIST input images.

## Sanity check

Placeholder for...:
* description
* results
* instructions how to repeat

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
