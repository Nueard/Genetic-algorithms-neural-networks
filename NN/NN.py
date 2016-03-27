import theano.tensor as T
import theano
import numpy

#### Main class used to construct and train networks
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

class NNet(object):
    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        self.validation_accuracies = []
        self.test_accuracies = []
        self.confusion_matrix = numpy.zeros((self.layers[-1].n_out,self.layers[-1].n_out))

    def display_weights(self):
        self.layers[0].display_weights()

    def train(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0, momentum=0.9):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        num_test_batches = int(size(test_data)/mini_batch_size)

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers if layer.w is not None])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        best_iteration = 0
        test_accuracy = 0
        # train
        confusion_matrix = numpy.zeros((self.layers[-1].n_out,self.layers[-1].n_out))
        for epoch in range(epochs):
            # train
            for minibatch_index in range(num_training_batches):
                cost_ij = train_mb(minibatch_index)

            # validate
            accuracies = []
            for j in range(num_validation_batches):
                expected_output, output = validate_mb_accuracy(j)
                accuracies.append(numpy.mean(numpy.equal(expected_output, output)))
            validation_accuracy = numpy.mean(accuracies)
            self.validation_accuracies.append(validation_accuracy)
            print(str(epoch) + " " + str(validation_accuracy))

            # test
            mb_test_accuracy = []
            tests = 0
            confusion_matrix = numpy.zeros((self.layers[-1].n_out,self.layers[-1].n_out))
            for j in range(num_test_batches):
                expected_output, output = test_mb_accuracy(j)
                tests += len(output)
                for k in range(len(output)):
                    confusion_matrix[expected_output[k],output[k]] += 1

                mb_test_accuracy.append(numpy.mean(numpy.equal(expected_output, output)))
            self.test_accuracies.append(numpy.mean(mb_test_accuracy))

        confusion_matrix = confusion_matrix.T
        numpy.savetxt('confusion.txt', confusion_matrix, delimiter=",")



        return [self.validation_accuracies, self.test_accuracies]
