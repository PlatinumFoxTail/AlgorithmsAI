import pickle
import gzip
import numpy as np

#training_data is tuples of 50 000pcs MNIST data. First tuple value contains MNIST training images and second tuple value is the digit value (0-9) of... 
#corresponding image. Each image is a numpy ndarray with 784 values (28 x 28 pixels). validation_data and test_data are similar to training_data, but... 
#contains only 10 000images each.
def load_data():
    #opening gzip-compressed file in binary read mode
    f = gzip.open('mnist.pkl.gz', 'rb')

    #deserialize the data from the file with pickel
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()

    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()

    #first tuple element x is 784-dimensional NumPy array representing images
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]

    #second tuple element y is 10-dimensional NumPy array i.e. image digit value
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    #second tuple element y is an integer i.e. image digit value
    validation_data = zip(validation_inputs, va_d[1])
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)

#unit vector to describe right digit in vector from i.e. certain index 1.0 and other indexes 0 is digit=index 
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e