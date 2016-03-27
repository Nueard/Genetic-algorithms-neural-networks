import theano
import theano.tensor as T
import sys
import numpy

def ReLU(z): return T.maximum(0.0, z)

from NN.NN import NNet
from NN.Layers.FullyConnected import FullyConnectedLayer
from NN.Layers.Convolutional import ConvolutionalLayer
from NN.Layers.Pooling import PoolingLayer
from NN.Layers.Softmax import SoftmaxLayer
from NN.DataLoader import load_data

# CONFIG
try: theano.config.device = 'gpu'
except: pass
theano.config.floatX = 'float32'
numpy.random.seed(0)

# HYPER PARAMETERS
mini_batch_size = 100
epochs = 200
learning_rate = 0.0001
l2=0.01

net = NNet([
            ConvolutionalLayer(image_shape=(mini_batch_size, 3, 128, 128),
                      filter_shape=(100, 3, 9, 9), activation_fn=ReLU),
            PoolingLayer(image_shape=(mini_batch_size, 100, 128, 128),
                        poolsize=(3,3)),

            ConvolutionalLayer(image_shape=(mini_batch_size, 100, 42, 42),
                      filter_shape=(50, 100, 2, 2), activation_fn=ReLU),
            PoolingLayer(image_shape=(mini_batch_size, 50, 42, 42),
                        poolsize=(3,3)),

            ConvolutionalLayer(image_shape=(mini_batch_size, 50, 14, 14),
                      filter_shape=(10, 50, 2, 2), activation_fn=ReLU),
            PoolingLayer(image_shape=(mini_batch_size,10, 14, 14),
                        poolsize=(2,2)),

            FullyConnectedLayer(n_in=10*7*7, n_out=100, activation_fn=ReLU, p_dropout=0.5),
            FullyConnectedLayer(n_in=100, n_out=10, activation_fn=ReLU, p_dropout=0.5),

            SoftmaxLayer(n_in=10, n_out=10)],
            mini_batch_size)

training_data, validation_data, test_data = load_data()

net.train(training_data, epochs, mini_batch_size, learning_rate,
            validation_data, test_data, lmbda=l2)
