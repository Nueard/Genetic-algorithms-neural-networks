from theano.tensor.nnet import sigmoid
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

import theano
import numpy
import skimage

class PoolingLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, image_shape, poolsize=(2, 2)):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.poolsize = poolsize
        self.image_shape = image_shape
        self.w = None
        self.params = []

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        self.output = pool.pool_2d(
            input=self.inpt, ds=self.poolsize, ignore_border=True)
        self.output_dropout = self.output # no dropout in the convolutional layers

    def get_output_shape(self):
        size = int(self.image_shape[2]/self.poolsize[0])
        return [self.image_shape[1], size, size]

    def validate(self):
        if numpy.prod(self.image_shape) == 0:
            return False
        if self.image_shape[2] < self.poolsize[0]:
            return False
        return True
