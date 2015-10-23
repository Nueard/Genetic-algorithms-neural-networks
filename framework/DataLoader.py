import numpy
import theano
import glob
import skimage.io
import theano.tensor as T

import sys
import gc

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
training data is 90% of loaded data
validation data is 5% of loaded data
test data is 5% of loaded data

"""
def loadData(path="../data/images/", dataSplit = [1,0.1,0.1]):

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

    gc.collect()
    if len(data) == 0:
        raise Exception("There is problem loading the data from " + path)

    data = numpy.transpose(data.tolist())
    numpy.random.shuffle(data.tolist())
    data = numpy.transpose(data.tolist())

    training_data = data
    validation_data = data[0:data.shape[0]/10,:]
    test_data = data = data[data.shape[0]/10:2*data.shape[0]/10,:]
    print "(debug) Number of training images: " + str(training_data.shape[0])
    print "(debug) Number of validation images: " + str(validation_data.shape[0])
    print "(debug) Number of test images: " + str(test_data.shape[0])

    return [shared(training_data), shared(validation_data), shared(test_data)]
