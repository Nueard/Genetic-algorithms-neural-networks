from framework.Network import Network
from framework.ConvPoolLayer import ConvPoolLayer
from framework.FullyConnectedLayer import FullyConnectedLayer
from framework.SoftmaxLayer import SoftmaxLayer
from framework.DataLoader import loadData
import theano.tensor as T

def ReLU(z): return T.maximum(0.0, z)

import theano

import sys
import numpy
import time

def enableGpu(flag=None):
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

    enableGpu()

    print "... setting up the network"

    mini_batch_size = 10


    net = Network([
                ConvPoolLayer(image_shape=(mini_batch_size, 3, 256, 256),
                          filter_shape=(100, 3, 7, 7),
                          poolsize=(5, 5)),
                # output ( 256 - 7 + 1 ) / 5 = 50
                ConvPoolLayer(image_shape=(mini_batch_size, 100, 50, 50),
                          filter_shape=(40, 100, 3, 3),
                          poolsize=(4, 4)),
                # output ( 50 - 3 + 1 ) / 4 = 12
                FullyConnectedLayer(n_in=40*12*12, n_out=1000, p_dropout=0.2),
                FullyConnectedLayer(n_in=1000, n_out=100, p_dropout=0.2),
                SoftmaxLayer(n_in=100, n_out=8, p_dropout=0.2)],
                mini_batch_size)

    print "... loading data"

    training_data, validation_data, test_data = loadData()

    print "... starting network training"
    start = time.time()
    net.train(training_data, 400, mini_batch_size, 0.01,
                validation_data, test_data)

    end = time.time()
    print "Training time elapsed: " + str(end - start) + " s"

    # net.display_weights()
