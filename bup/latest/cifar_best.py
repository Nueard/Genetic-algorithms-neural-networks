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
from NN.CifarLoader import load_data

# CONFIG
try: theano.config.device = 'gpu'
except: pass
theano.config.floatX = 'float32'
numpy.random.seed(0)

# HYPER PARAMETERS
mini_batch_size = 100
epochs = 70
learning_rate = 0.01
l2=0.01

net = NNet([
            ConvolutionalLayer(image_shape=(mini_batch_size, 3, 32, 32),
                      filter_shape=(64, 3, 3, 3), activation_fn=ReLU),
            ConvolutionalLayer(image_shape=(mini_batch_size, 64, 32, 32),
                      filter_shape=(64, 64, 3, 3), activation_fn=ReLU),
            PoolingLayer(image_shape=(mini_batch_size, 64, 32, 32),
                        poolsize=(2,2)),

            ConvolutionalLayer(image_shape=(mini_batch_size, 64, 16, 16),
                      filter_shape=(128, 64, 3, 3), activation_fn=ReLU),
            ConvolutionalLayer(image_shape=(mini_batch_size, 128, 16, 16),
                      filter_shape=(128, 128, 3, 3), activation_fn=ReLU),
            PoolingLayer(image_shape=(mini_batch_size, 128, 16, 16),
                        poolsize=(2,2)),

            ConvolutionalLayer(image_shape=(mini_batch_size, 128, 8, 8),
                      filter_shape=(256, 128, 3, 3), activation_fn=ReLU),
            ConvolutionalLayer(image_shape=(mini_batch_size, 256, 8, 8),
                      filter_shape=(256, 256, 3, 3), activation_fn=ReLU),
            ConvolutionalLayer(image_shape=(mini_batch_size, 256, 8, 8),
                      filter_shape=(256, 256, 3, 3), activation_fn=ReLU),
            ConvolutionalLayer(image_shape=(mini_batch_size, 256, 8, 8),
                      filter_shape=(256, 256, 3, 3), activation_fn=ReLU),
            PoolingLayer(image_shape=(mini_batch_size,256, 8, 8),
                        poolsize=(2,2)),

            FullyConnectedLayer(n_in=256*4*4, n_out=1024, activation_fn=ReLU, p_dropout=0.5),
            FullyConnectedLayer(n_in=1024, n_out=1024, activation_fn=ReLU, p_dropout=0.5),
            FullyConnectedLayer(n_in=1024, n_out=10, activation_fn=ReLU, p_dropout=0.5),

            SoftmaxLayer(n_in=10, n_out=10)],
            mini_batch_size)

training_data, validation_data, test_data = load_data()

net.train(training_data, epochs, mini_batch_size, learning_rate,
            validation_data, test_data, lmbda=l2)
