import Network
import ConvPoolLayer
import HiddenLayer
import SoftmaxLayer
from DataLoader import loadData

def enableGpu(flag):
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'

def unsupportedFlag(flag):
    print "Unsupported flag: " + flag + " . Please see documentation"+ \
    "for more information."

args = {
    "-gpu" : enableGpu
}

if __name__ == '__main__':
    for arg in sys.argv:
        if arg in args:
            args[arg](arg)
        elif arg.find('-') == 0:
            unsupportedFlag(arg)

    training_data, validation_data, test_data = loadData()
    mini_batch_size = 10

    net = Network([
            FullyConnectedLayer(n_in=65536, n_out=10000),
            FullyConnectedLayer(n_in=10000, n_out=1000),
            SoftmaxLayer(n_in=1000, n_out=500),
            FullyConnectedLayer(n_in=500, n_out=8)],
            mini_batch_size)

    net.train(training_data, 60, mini_batch_size, 0.1,
                validation_data, test_data)
