# IPL4
Genetic algorithm with neural networks frameworks developed for my level 4 individual project.

# Installation instructions
There are several components needed in order to run this project:

Numpy - to install numpy please visit [link to Numpy install guide](http://www.scipy.org/install.html)
Theano - to install theano please visit [link to Theano install guide](http://deeplearning.net/software/theano/install.html)

If you have a NVIDIA graphics card you will need to setup CUDA. You can get it from [here](https://developer.nvidia.com/cuda-downloads).
If you are a registered NVIDIA developer you might also want to install NVIDIA CUDA Deep Neural Network (cuDNN). It will speed up the training process of neural networks. You can download cuDNN from [here](https://developer.nvidia.com/cudnn)

# Running instructions
There are three different frameworks in this project. A deep convolutional neural networks framework, a genetic algorithms framework and a combination of both - genetic algorithms with neural networks framework.
The first one can be used as specified in '<nn.py>' file.

## Neural networks framework
The basic usage is:
* Define the network.

```
net = NNet([
            ConvolutionalLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5), activation_fn=ReLU),
            PoolingLayer(image_shape=(mini_batch_size,20, 28, 28),
                        poolsize=(2,2)),
            ConvolutionalLayer(image_shape=(mini_batch_size, 20, 14, 14),
                      filter_shape=(40, 20, 5, 5), activation_fn=ReLU),
            PoolingLayer(image_shape=(mini_batch_size, 40, 14, 14),
                        poolsize=(2,2)),
            FullyConnectedLayer(n_in=40*7*7, n_out=1000, activation_fn=ReLU),
            FullyConnectedLayer(n_in=1000, n_out=1000, activation_fn=ReLU),
            SoftmaxLayer(n_in=1000, n_out=10)],
            mini_batch_size)
```

* Load data needed to train, validate and test the network.

```
training_data, validation_data, test_data = load_data()
```

* Train the network.
```
net.train(training_data, epochs, mini_batch_size, learning_rate,
            validation_data, test_data, lmbda=l2)
```

## Genetic algorithms framework
The genetic algorithms framework is very abstract. There isn't an example since we don't use the pure version of it, rather than that we use a mixed version of neural networks and genetic algorithms.

## Genetic algorithms with neural networks framework
There is an example file that shows how to run the GANN framework - 'gann.py'. There are several parameters you need to specify as discussed in the dissertation.

```
parameters["ga"] = {
    "maxGenerations": 1000,
    "threshold": 0.9,
    "population": 20,
    "crossover": 0.2,
    "mutation": 0.1
}

parameters["nn"] = {
    "params" : {
        "layers": [4,10],
        "batchSize": 100,
        "maxEpochs": 5,
        "inputShape": [3,32,32],
        "output": 10,
        "data_split":[0.5, 0.75, 1],
        "learningRate": [0,2],
        "l2": [0,2]
    },
    "definitions" : {
        "convolution": {
            "filters": [1,32],
            "size": [1,8]
        },
        "pooling": {
            "size": [1,4]
        },
        "fullyconnected":{
            "neurons": [1,1000],
            "dropout": [0, 0.5]
        },
        "softmax":{
            "dropout": [0, 0.5]
        }
    }
}
```

After you define those, you need to initialise the GANN framework by providing those parameters to the constructor, as well as the 'load_data' function you want to use.
There are several 'load_data' functions provided for the MNIST, CIFAR and larger images data sets. They can be imported like this

```
# choose one of the datasets
from NN.CIFARLoader import load_data # CIFAR
from NN.DataLoader import load_data # Larger images
from NN.MNISTLoader import load_data # MNIST

# initialise GANN framework
g = GANN(parameters, load_data, 'test.obj')
```

After initialising the GANN framework we need to run it by

```
ga.run()
```
