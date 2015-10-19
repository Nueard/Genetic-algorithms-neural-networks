import network
from network import Network
from network import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

training_data, validation_data, test_data = network.load_data_shared()
mini_batch_size = 10
print "Done loading data"
net = Network([
        FullyConnectedLayer(n_in=65536, n_out=10000),
        FullyConnectedLayer(n_in=10000, n_out=1000),
        SoftmaxLayer(n_in=1000, n_out=500),
        FullyConnectedLayer(n_in=500, n_out=8)],
        mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1,
            validation_data, test_data)
