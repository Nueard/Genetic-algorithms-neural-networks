import theano
import theano.tensor as T
import sys
import numpy
# define ReLU activation function
def ReLU(z): return T.maximum(0.0, z)

from NN.NN import NNet
from NN.Layers.FullyConnected import FullyConnectedLayer
from NN.Layers.Convolutional import ConvolutionalLayer
from NN.Layers.Pooling import PoolingLayer
from NN.Layers.Softmax import SoftmaxLayer

# choose dataset
# from NN.CIFARLoader import load_data # CIFAR
# from NN.DataLoader import load_data # Larger images
from NN.MNISTLoader import load_data # MNIST

# config GPU
try: theano.config.device = 'gpu'
except: pass
theano.config.floatX = 'float32'
numpy.random.seed(0)

# hyper parameters
mini_batch_size = 100
epochs = 60
learning_rate = 0.03
l2 = 0.1

# network definition
net = NNet([
            ConvolutionalLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5), activation_fn=ReLU),
            PoolingLayer(image_shape=(mini_batch_size,20, 28, 28),
                        poolsize=(2,2)),
            ConvolutionalLayer(image_shape=(mini_batch_size, 20, 14, 14),
                      filter_shape=(40, 20, 5, 5), activation_fn=ReLU),
            PoolingLayer(image_shape=(mini_batch_size, 40, 14, 14),
                        poolsize=(2,2)),
            FullyConnectedLayer(n_in=40*7*7, n_out=1000, activation_fn=ReLU),
            FullyConnectedLayer(n_in=1000, n_out=1000, activation_fn=ReLU),
            SoftmaxLayer(n_in=1000, n_out=10)],
            mini_batch_size)

# load data
training_data, validation_data, test_data = load_data()

# train
net.train(training_data, epochs, mini_batch_size, learning_rate,
            validation_data, test_data, lmbda=l2)
