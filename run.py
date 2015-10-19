from framework.Network import Network
from framework.ConvPoolLayer import ConvPoolLayer
from framework.FullyConnectedLayer import FullyConnectedLayer
from framework.SoftmaxLayer import SoftmaxLayer
from framework.DataLoader import loadData

import sys

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
    "-gpu" : enableGpu,
    "-test" : test,
    "-shorttest" : test,
    "-realdata" : test
}

if __name__ == '__main__':
    for arg in sys.argv:
        if arg in args:
            args[arg](arg)
        elif arg.find('-') == 0:
            unsupportedFlag(arg)

    training_data, validation_data, test_data = loadData(data)
    mini_batch_size = 10

    print "... setting up the network"

    net = Network([
            FullyConnectedLayer(n_in=65536, n_out=20000),
            FullyConnectedLayer(n_in=20000, n_out=1000),
            # FullyConnectedLayer(n_in=1000, n_out=100),
            SoftmaxLayer(n_in=1000, n_out=8)],
            mini_batch_size)

    print "... starting network training"

    net.train(training_data, 60, mini_batch_size, 0.1,
                validation_data, test_data)