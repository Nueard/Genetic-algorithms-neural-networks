from framework.Network import Network
from framework.ConvPoolLayer import ConvPoolLayer
from framework.FullyConnectedLayer import FullyConnectedLayer
from framework.SoftmaxLayer import SoftmaxLayer
from framework.DataLoader import loadData, load_data_mnist
import theano.tensor as T
def ReLU(z): return T.maximum(0.0, z)

import theano

import sys
import numpy
import time

def enableGpu(flag):
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'

def unsupportedFlag(flag):
    print "Unsupported flag: " + flag + " . Please see documentation"+ \
    "for more information."

data = ""

def test(type):
    global data
    if type == "-shorttest":
        data = "data/sample/"
    elif type == "-test":
        data = "data/large_sample/"
    elif type == "-realdata":
        data = "data/images/"

args = {
    "-test" : test,
    "-shorttest" : test,
    "-realdata" : test
}

if __name__ == '__main__':
    numpy.random.seed(0)
    for arg in sys.argv:
        if arg in args:
            args[arg](arg)
        elif arg.find('-') == 0:
            unsupportedFlag(arg)

    enableGpu(None)
    training_data, validation_data, test_data = load_data_mnist()
    mini_batch_size = 10

    print "... setting up the network"

    net = Network([
            ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                          filter_shape=(20, 1, 5, 5),
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                          filter_shape=(40, 20, 5, 5),
                          poolsize=(2, 2),
                          activation_fn=ReLU),
            FullyConnectedLayer(
                n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
            FullyConnectedLayer(
                n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
            SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
            mini_batch_size)


                net = Network([
                        ConvPoolLayer(image_shape=(mini_batch_size, 3, 256, 256),
                                  filter_shape=(100, 3, 17, 17),
                                  poolsize=(4, 4)),
                        # output ( 256 - 17 + 1 ) / 4 = 60
                        ConvPoolLayer(image_shape=(mini_batch_size, 100, 60, 60),
                                  filter_shape=(40, 100, 5, 5),
                                  poolsize=(4, 4)),
                        # output ( 60 - 5 + 1 ) / 4 = 14
                        ConvPoolLayer(image_shape=(mini_batch_size, 40, 14, 14),
                                filter_shape=(15, 40, 5, 5),
                                poolsize=(2, 2)),
                        # output ( 14 - 5 + 1 ) / 2 = 5
                        FullyConnectedLayer(n_in=15*5*5, n_out=100, p_dropout=0.1),
                        SoftmaxLayer(n_in=100, n_out=8, p_dropout=0.1)],
                        mini_batch_size)

    print "... starting network training"

    start = time.time()
    net.train(training_data, 40, mini_batch_size, 0.05,
                validation_data, test_data)

    end = time.time()
    print "Training time elapsed: " + str(end - start) + " s"

    # net.display_weights()
