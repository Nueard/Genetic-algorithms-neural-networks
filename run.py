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
                          filter_shape=(100, 3, 11, 11),
                          poolsize=(6, 6)),
                # output ( 256 - 11 + 1 ) / 6 = 41
                ConvPoolLayer(image_shape=(mini_batch_size, 100, 41, 41),
                          filter_shape=(40, 100, 3, 3),
                          poolsize=(3, 3)),
                # output ( 41 - 3 + 1 ) / 3 = 13
                ConvPoolLayer(image_shape=(mini_batch_size, 40, 13, 13),
                          filter_shape=(10, 40, 2, 2),
                          poolsize=(2, 2)),
                # output ( 13 - 2 + 1 ) / 2 = 6
                FullyConnectedLayer(n_in=10*6*6, n_out=80, p_dropout=0.2),
                SoftmaxLayer(n_in=80, n_out=8, p_dropout=0.2)],
                mini_batch_size)

    print "... loading data"

    training_data, validation_data, test_data = loadData()

    print "... starting network training"
    start = time.time()
    net.train(training_data, 400, mini_batch_size, 0.05,
                validation_data, test_data)

    end = time.time()
    print "Training time elapsed: " + str(end - start) + " s"

    # net.display_weights()
