import theano
import sys
import numpy

from NN.NN import NNet
from NN.Layers.Convolutional import ConvolutionalLayer
from NN.Layers.Pooling import PoolingLayer
from NN.Layers.FullyConnected import FullyConnectedLayer
from NN.Layers.Softmax import SoftmaxLayer

# CONFIG
try: theano.config.device = 'gpu'
except: pass
theano.config.floatX = 'float32'
numpy.random.seed(0)

# HYPER PARAMETERS
mini_batch_size = 100
epochs = 2000
learning_rate = 0.03
data_split = [0.5, 0.75, 1]

net = NNet([
    ConvolutionalLayer(image_shape=(mini_batch_size, 3, 128, 128),
                filter_shape=(100, 3, 9, 9)),
    ConvolutionalLayer(image_shape=(mini_batch_size, 3, 128, 128),
                filter_shape=(100, 3, 9, 9),
                poolsize=(3, 3)),
    # output ( 128 - 9 + 1 ) / 3 = 40
    ConvPoolLayer(image_shape=(mini_batch_size, 100, 40, 40),
                filter_shape=(50, 100, 2, 2),
                poolsize=(3, 3)),
    # output ( 40 - 2 + 1 ) / 3 = 13
    ConvPoolLayer(image_shape=(mini_batch_size, 50, 13, 13),
                filter_shape=(10, 50, 2, 2),
                poolsize=(2, 2)),
    # output ( 13 - 2 + 1 ) / 2 = 6
    FullyConnectedLayer(n_in=10*6*6, n_out=80, p_dropout=0.2),
    SoftmaxLayer(n_in=80, n_out=8, p_dropout=0.2)],
    mini_batch_size)

training_data, validation_data, test_data = load_data(data_split=data_split)

net.train(training_data, epochs, mini_batch_size, learning_rate,
    validation_data, test_data)
