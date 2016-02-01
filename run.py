import random
import theano
import sys
import numpy

from GA.GA import GAlg
from NN.NN import NNet
from NN.Layers.Convolutional import ConvolutionalLayer
from NN.Layers.Pooling import PoolingLayer
from NN.Layers.FullyConnected import FullyConnectedLayer
from NN.Layers.Softmax import SoftmaxLayer
from NN.CifarLoader import load_data

params = {
    "layers": [2,5],
    "batchSize": 100,
    "maxEpochs": 10,
    "inputShape": [3,32,32],
    "output": 8,
    "data_split":[0.5, 0.75, 1]
}

def convolution(inpt, this):
    if len(inpt) != 3:
        return None
    this["filters"] = int(this["filters"])
    this["size"] = int(this["size"])
    image_shape = (params["batchSize"], inpt[0], inpt[1], inpt[2])
    filter_shape = (this["filters"], inpt[0], this["size"], this["size"])
    return ConvolutionalLayer(image_shape, filter_shape)

def pooling(inpt, this):
    if len(inpt) != 3:
        return None
    this["size"] = int(this["size"])
    image_shape = (params["batchSize"], inpt[0], inpt[1], inpt[2])
    poolsize = (this["size"], this["size"])
    return PoolingLayer(image_shape, poolsize)

def fullyconnected(inpt, this):
    this["neurons"] = int(this["neurons"])
    inpt = int(numpy.prod(inpt))
    return FullyConnectedLayer(inpt, this["neurons"], this["dropout"])

def softmax(inpt, this):
    this["neurons"] = int(this["neurons"])
    inpt = int(numpy.prod(inpt))
    return SoftmaxLayer(inpt, this["neurons"], this["dropout"])

definitions = {
    "convolution": {
        "filters": [1,32],
        "size": [1,8],
        "fn": convolution
    },
    "pooling": {
        "size": [1,4],
        "fn": pooling
    },
    "fullyconnected":{
        "neurons": [1,1000],
        "dropout": [0, 0.2],
        "fn": fullyconnected
    },
    "softmax":{
        "neurons": [params["output"],params["output"]],
        "dropout": [0, 0.2],
        "fn": softmax
    }
}

ga_params = {
    "maxGenerations": 10000,
    "threshold": 0.9,
    "population": 10,
    "crossover": 0.2,
    "mutation": 1
}

def defineLayer(inpt,this):
    layer = this["fn"](inpt, this)
    if layer != None and layer.validate():
        return layer
    else:
        return None

def generateLayer(typ):
    layer = {"type": typ}
    for key in definitions[typ].keys():
        if key == "fn":
            layer[key] = definitions[typ][key]
            continue
        layer[key] = random.uniform(definitions[typ][key][0],definitions[typ][key][1])
    return layer

def generate():
    numLayers = random.randint(params["layers"][0],params["layers"][1])
    layers = []

    inpt = params["inputShape"]
    layer = generateLayer("convolution")
    layers.append(layer)

    for i in range(numLayers):
        k = list(definitions.keys())
        k.remove("softmax")
        typ = random.choice(k)
        layer = generateLayer(typ)
        layers.append(layer)

    layer = generateLayer("softmax")
    layers.append(layer)

    return layers

def crossover(one, two):
    pointOne = random.randint(1,len(one)-2)
    pointTwo = random.randint(1,len(two)-2)
    resOne = one[:pointOne] + two[pointTwo:]
    resTwo = one[pointOne:] + two[:pointTwo]

    print(resOne[:].keys())
    print(resTwo[:].keys())
    return resOne, resTwo

def mutate(ind):
    print(len(ind))
    index = random.randint(0,len(ind)-1)
    key = "fn"
    forbidden = ["fn", "type"]
    while key in forbidden:
        key = random.choice(list(ind[index].keys()))

    typ = ind[index]["type"]
    new_value = int(random.uniform(definitions[typ][key][0],definitions[typ][key][1]))
    ind[index][key] = new_value
    return ind

def evaluate(layers):
    dfns = []
    for i in range(len(layers)):
        inpt = None
        if len(dfns) == 0:
            inpt = params["inputShape"]
        else:
            inpt = dfns[-1].get_output_shape()
        dfn = defineLayer(inpt, layers[i])
        if dfn != None:
            dfns.append(dfn)
    net = NNet(dfns, params["batchSize"])
    fitness = net.train(training_data, params["maxEpochs"], params["batchSize"], 0.02,
        validation_data, test_data)
    print("Individual fitness " + str(fitness))
    return fitness




# CONFIG
try: theano.config.device = 'gpu'
except: pass
theano.config.floatX = 'float32'
numpy.random.seed(0)
training_data, validation_data, test_data = load_data(data_split=params["data_split"])

fns = {
    "generate": generate,
    "evaluate": evaluate,
    "mutate": mutate,
    "crossover": crossover
}
dump_file = "dump.obj"
if len(sys.argv) > 1:
    if sys.argv[1] == "resume" and len(sys.argv) == 3:
        dump_file = sys.argv[2]

ga = GAlg(ga_params, fns, dump_file)
ga.run()
