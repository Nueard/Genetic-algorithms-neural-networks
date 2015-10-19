import numpy
import theano
import glob
import skimage.io
import theano.tensor as T

""" Returns the index (0 .. len(categories)) of the image by trying to find a
category in the filename, or -1 if not found

:type file: string
:param file: file name and path e.g. ../data/image/forest-1.jpg

"""
def get_index(file):
    categories = ['building', 'forest', 'mountain', 'country', 'city', 'street',
                    'highway', 'coast']
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
    data = numpy.empty([0,2], dtype=theano.config.floatX)
    for file in glob.glob(path+"*.jpg"):
        image = skimage.io.imread(file)
        gray = 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]
        data = numpy.vstack((data, [gray.flatten().tolist(), get_index(file)]))

    if len(data) == 0:
        raise Exception("There is problem loading the data from " + path)

    numpy.random.shuffle(data);
    B=numpy.random.randint(data.shape[0],size=int(data.shape[0]*dataSplit[0]))
    training_data = data[B,:]
    B=numpy.random.randint(data.shape[0],size=int(data.shape[0]*dataSplit[1]))
    validation_data = data[B,:]
    B=numpy.random.randint(data.shape[0],size=int(data.shape[0]*dataSplit[1]))
    test_data = data[B,:]

    print "... successfully loaded " + str(data.shape[0]) + " images"

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
                    numpy.asarray(data[:,0].tolist(), dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
                    numpy.asarray(data[:,1].tolist(), dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]