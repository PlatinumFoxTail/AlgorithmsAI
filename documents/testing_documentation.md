# Testing documentation

The testing focus of this project is set on the neural network (network.py), and not other parts such as the user interface (digit_predicter.ipynb). The testing can be divided into two parts: Unit testing & checkstyle and Sanity check. The former is self-explanatory, and the Sanity check includes a comparsion of intermediate outputs from the neural network and a simplified neural network based on simplified MNIST input images.

## Unit testing

The coverage as of 2nd of Feb is 73%. Codecov badge as well as pylint to be added after final unit tests are ready.

## Sanity check

Placholder for sanity check results. Idea/plan as of 2nd of Feb:

* Simplify the MNIST input images e.g. to 10pixels
* Set up a simplified network e.g. ready library
* Train the simplified network and own network with the simplified input
* Compare intermediate calculation steps between the simplified network and own network e.g. feedforward outputs.
