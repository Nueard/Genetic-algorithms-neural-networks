from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import conv

import theano
import numpy
import skimage

class ConvolutionalLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*numpy.prod(filter_shape[2:]))
        self.w = theano.shared(
            numpy.asarray(
                numpy.random.normal(loc=0, scale=numpy.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)

        self.b = theano.shared(
            numpy.asarray(
                numpy.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        self.output = self.activation_fn(
            conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

    def validate(self):
          if numpy.prod(self.image_shape) == 0:
              return False
          # filter is larger than image
          if self.image_shape[2] < self.filter_shape[2]:
              return False
          return True

    def get_output_shape(self):
          size = self.image_shape[2]-self.filter_shape[2]+1
          return [self.filter_shape[0], size, size]
