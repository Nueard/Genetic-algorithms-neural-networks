import theano
import sys
import numpy

from NN.NN import NNet
from NN.Layers.FullyConnected import FullyConnectedLayer
from NN.Layers.Softmax import SoftmaxLayer
from NN.MNISTLoader import load_data

# CONFIG
try: theano.config.device = 'gpu'
except: pass
theano.config.floatX = 'float32'
numpy.random.seed(0)

# HYPER PARAMETERS
mini_batch_size = 100
epochs = 100
learning_rate = 1

net = NNet([
            FullyConnectedLayer(n_in=784, n_out=2000),
            FullyConnectedLayer(n_in=2000, n_out=10),
            SoftmaxLayer(n_in=10, n_out=10)],
            mini_batch_size)

training_data, validation_data, test_data = load_data()

net.train(training_data, epochs, mini_batch_size, learning_rate,
            validation_data, test_data)
