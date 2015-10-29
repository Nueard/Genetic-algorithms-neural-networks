import numpy
import theano
import glob
import skimage.io
import theano.tensor as T

import sys
import gc
import cPickle
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
    return -1

""" Loads the test data in the specified directory. Reads all the files there
and organizes them in 2d array where
image_n = return[n,0]
category_of_image_n = return[n,1]

:type path: string
:param path: Filepath of folder with data e.g. /data/images/

:type dataSplit: array[int]
:param dataSplit: Array of floats 0..1 which represents fractions for training,
validation and test data. Example:
[0.9,0.05,0.05]
training data is 60% of loaded data
validation data is 20% of loaded data
test data is 20% of loaded data

"""
def loadData(path="./data/images/", data_split = [0.8,1,1]):

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
                    numpy.asarray(data[:,0].tolist(), dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
                    numpy.asarray(data[:,1].tolist(), dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")


    data = numpy.empty([0,2], dtype=theano.config.floatX)
    for index,file in enumerate(glob.glob(path+"*.jpg")):
        image = skimage.io.imread(file)
        data = numpy.vstack((data, [image.flatten(), get_index(file)]))

    data_split[0] = int(data.shape[0]*data_split[0])
    data_split[1] = int(data.shape[0]*data_split[1])
    data_split[2] = int(data.shape[0]*data_split[2])

    gc.collect()
    if len(data) == 0:
        raise Exception("There is problem loading the data from " + path)

    data = numpy.transpose(data.tolist())
    numpy.random.shuffle(data.tolist())
    data = numpy.transpose(data.tolist())

    training_data = data[:data_split[0],:]
    validation_data = data[data_split[0]:data_split[1],:]
    # test_data = data = data[data_split[1]:data_split[2],:]
    print "(debug) Number of training images: " + str(training_data.shape[0])
    print "(debug) Number of validation images: " + str(validation_data.shape[0])
    # print "(debug) Number of test images: " + str(test_data.shape[0])

    # return [shared(training_data), shared(validation_data), shared(test_data)]
    return [shared(training_data), shared(validation_data), None]

def load_data_mnist(filename="./data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            numpy.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            numpy.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]
