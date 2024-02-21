import numpy as np
from network import Network
from simple_network import SimpleNetwork
import matplotlib.pyplot as plt

#setting simple input data
input = np.array([[0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1]])

#setting simple output data in binary form, so that 1 neuron in output layer will be able to classify
output = np.array([[1], [0], [0], [1]])

#training network
def train_network(input, output):
    #3 input neurons, 4 hidden neuons, 1 output neuron
    network = Network([3, 4, 1])
    #reshape to column vectors and make tuples of vector elements
    training_data = [(input.reshape(-1, 1), output.reshape(-1, 1)) for input, output in zip(input, output)]

    hidden_layer_outputs = []
    output_layer_outputs = []

    #30 000 iterations training
    for _ in range(30000):
        for mini_batch in training_data:
            #updating weights and biases, with learning rate 0.01
            network.update_mini_batch([mini_batch], 0.01)

        #storing output values per iteration
        outputs = network.feedforward(input.T)
        #only selecting value corresponding to hidden layer's four neurons.
        #flatten converts to 1-D array and tolist converts to Python list for plotting purpouses
        hidden_layer_outputs.append(outputs[0].flatten().tolist())
        #only selecting value corresponding to output layer's only neuron
        output_layer_outputs.append(outputs[-1][0])

    return hidden_layer_outputs, output_layer_outputs

#training simple_network
def train_simple_network(input, output):
    simple_network = SimpleNetwork(input, output)
    #learning rate 0.01 as for network
    hidden_layer_outputs, output_layer_outputs = simple_network.train(0.01)

    return hidden_layer_outputs, output_layer_outputs

print("network being trained...")
#hlo, hidden layer outputs. olo, output layer outputs
hlo_network, olo_network = train_network(input, output)
print("\nsimple_network being trained...")
hlo_simple_network, olo_simple_network = train_simple_network(input, output)

plt.figure(figsize=(15, 5))
plt.suptitle('network vs. simple_network. eta=0.01, no biases')
plt.tight_layout()

#plotting hidden layer neurons' outputs of network and simple_network
plt.subplot(1, 2, 1)
for i in range(len(hlo_network[0])):
    #plotting every 1000th value
    plt.plot(
        range(0, len(hlo_network), 1000),
        [output[i] for output in hlo_network][::1000],
        'x',
        label=f'network, Neuron {i+1}'
        )
for i in range(len(hlo_simple_network[0])):
    #plotting every 1000th value
    plt.plot(
        range(0, len(hlo_simple_network), 1000),
        [output[i] for output in hlo_simple_network][::1000],
        'o',
        label=f'simple_network, Neuron {i+1}'
        )
plt.xlabel('Iterations')
plt.ylabel('Output')
plt.title('Hidden Layer Outputs')
plt.legend()

#plotting output layer neurons' outputs of network and simple_network
plt.subplot(1, 2, 2)
#plotting every 1000th value
plt.plot(range(0, len(olo_network), 1000), olo_network[::1000], 'x', label='network, Neuron 1')
#plotting every 1000th value
plt.plot(
    range(0, len(olo_simple_network), 1000),
    olo_simple_network[::1000],
    'o',
    label='simple_network, Neuron 1'
    )
plt.xlabel('Iterations')
plt.ylabel('Output')
plt.title('Output Layer Outputs')
plt.legend()

plt.show()