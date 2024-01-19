# Specifications

This document specifies the Datastructures and Algorithms Lab course at University of Helsinki, period III 2024.

## Topic

The topic of the project is a program that predicts handwritten digits. The user can choose handwitten digits from a specific dataset (MNIST) to be evaluated by the program. Both the chosen handwritten digits by the user and the program predicted digits will be visible for the the user for comparsion. The user will be able to run the program in a stepwise interactive way, so that beside the digit prediction the functionality of the program will also be presented. Beside implementing the program the program will be tested and documented as well.

The intention is to build a __multilayer perceptron (MLP) artificial neural network__ that will predict the value of the given handwritten digits. The MLP will be built from scratch i.e. not utilizing ready made MLP models such as PyTorch or scikit. The MLP was chosen, because i) its ability to predict images, ii) potential further build-on for more complex artificial neural networks such as convolutional neural network, and iii) its expected workload matches the width of the course.

The MNIST dataset was chosen as suitable starting point prior to analysis of more complex images. MNIST is a widely used benchmark data set in the field of machine learning and computer vision.  

## Programming languages

The program will be based on Python, and the user can run the program via Jupyter Notebook. Supportive libraries, such as Numpy-library for matrix operations, will be used.

Python unittest will be used for unit testing and Codecov for test coverage.

## Algorithm

The main sources for multilayer perceptron artificial network is still to be specified, but a starting point at least:

* [Wikipedia, Multilayer perceptron ](https://en.wikipedia.org/wiki/Multilayer_perceptron)

## Documentation

All the documentation and programming will be done in English.
