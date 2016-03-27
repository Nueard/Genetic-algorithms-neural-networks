# choose one of the datasets
# from NN.CIFARLoader import load_data # CIFAR
# from NN.DataLoader import load_data # Larger images
from NN.MNISTLoader import load_data # MNIST

from GANN.GANN import GANN

# define parameters
parameters = {}
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

def processData(generations):
    print(len(generations))

# initialise GANN framework
g = GANN(parameters, load_data, 'test.obj', processData)

# run
g.run()
