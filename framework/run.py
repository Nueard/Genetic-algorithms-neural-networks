import Network
import ConvPoolLayer
import HiddenLayer
import SoftmaxLayer
from DataLoader import loadData
import time

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
    print "... sleeping"
    time.sleep(180)    # pause 5.5 seconds
    #
    # net = Network([
    #         ConvPoolLayer(image_shape=(mini_batch_size, 1, 512, 512),
    #                   filter_shape=(20, 1, 5, 5),
    #                   poolsize=(4, 4)),
    #         # output is (512 - 5 + 1) / 4 = 127
    #         ConvPoolLayer(image_shape=(mini_batch_size, 20, 127, 127),
    #                   filter_shape=(40, 20, 4, 4),
    #                   poolsize=(2, 2)),
    #         # output is (127 - 4 + 1) / 2 = 62
    #         FullyConnectedLayer(n_in=62*62, n_out=100),
    #         SoftmaxLayer(n_in=100, n_out=8),
    #         mini_batch_size)
    #
    # net.train(training_data, 60, mini_batch_size, 0.1,
    #             validation_data, test_data)
