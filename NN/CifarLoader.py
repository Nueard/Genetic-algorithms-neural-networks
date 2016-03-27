import numpy
import theano
import pickle
import glob
import skimage.io
import theano.tensor as T

def load_data(path="./data/sample/", data_split = [0.8,0.9,1]):
    print ("... loading data")

    def unpickle(file):
        fo = open(file, 'rb')
        dict = pickle.load(fo,encoding='latin1')
        fo.close()
        return dict

    def shared(data):
        """ Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(numpy.asarray(data["data"].tolist(), dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(numpy.asarray(data["labels"], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")


    training_data = unpickle("./data/cifar/data_batch_1")
    for i in range(2,5):
        data = unpickle("./data/cifar/data_batch_"+str(i))
        training_data["data"] = numpy.append(training_data["data"],data["data"], axis=0)
        training_data["labels"].extend(data["labels"])

    validation_data = unpickle("./data/cifar/data_batch_5")
    test_data = unpickle("./data/cifar/test_batch")

    print ("(debug) Number of training images: " + str(training_data["data"].shape))
    print ("(debug) Number of validation images: " + str(validation_data["data"].shape))
    print ("(debug) Number of test images: " + str(test_data["data"].shape))

    return [shared(training_data), shared(validation_data), shared(test_data)]
