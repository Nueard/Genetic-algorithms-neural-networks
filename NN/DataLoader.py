import numpy
import theano
import glob
import skimage.io
import theano.tensor as T

import sys
import gc
import gzip

""" Returns the index (0 .. len(categories)) of the image by trying to find a
category in the filename, or -1 if not found

:type file: string
:param file: file name and path e.g. ../data/image/forest-1.jpg

"""
categories = ['building', 'forest', 'mountain', 'country', 'city', 'street',
                'highway', 'coast']

def get_index(file):
    for index,category in enumerate(categories):
        if file.find(category) != -1:
            return index
    print ("... index not found")
    return -1

def load_data(path="./data/images/", data_split = [0.8,0.9,1]):
    print ("... loading data")

    def shared(data):
        """ Place the data into shared variables. This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(numpy.asarray(data[:,0].tolist(), dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(numpy.asarray(data[:,1].tolist(), dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")


    data = numpy.ndarray((0,2))
    for index,file in enumerate(glob.glob(path+"*.jpg")):
        image = skimage.io.imread(file)
        data = numpy.vstack((data, [image.flatten().tolist(), get_index(file)]))

    data_split[0] = int(data.shape[0]*data_split[0])
    data_split[1] = int(data.shape[0]*data_split[1])
    data_split[2] = int(data.shape[0]*data_split[2])

    gc.collect()
    if len(data) == 0:
        raise Exception("There is problem loading the data from " + path)

    # Shuffle data
    data = data[numpy.random.permutation(data.shape[0]), :]

    training_data = data[:data_split[0],:]
    validation_data = data[data_split[0]:data_split[1],:]
    test_data = data[data_split[1]:data_split[2],:]
    print ("(debug) Number of training images: " + str(training_data.shape))
    print ("(debug) Number of validation images: " + str(validation_data.shape))
    print ("(debug) Number of test images: " + str(test_data.shape))

    return [shared(training_data), shared(validation_data), shared(test_data)]
