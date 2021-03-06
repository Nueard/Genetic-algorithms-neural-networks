from theano.tensor.nnet import sigmoid
import theano
import numpy
import theano.tensor as T
from theano.tensor import shared_randomstreams

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0, activation_fn=sigmoid):
        '''
            n_in is the number of inputs the fully connected layer will take,
            while n_out is the number of neurons the layer will have
        '''
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            numpy.asarray(
                numpy.random.normal(
                    loc=0.0, scale=numpy.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            numpy.asarray(numpy.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

    def validate(self):
        if self.n_out == 0:
            return False
        return True

    def get_output_shape(self):
        return [self.n_out]

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        numpy.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
