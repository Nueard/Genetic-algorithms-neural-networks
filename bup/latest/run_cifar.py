import random
import theano
import theano.tensor as T
import sys
import numpy

from GA.GA import GAlg
from NN.NN import NNet
from NN.Layers.Convolutional import ConvolutionalLayer
from NN.Layers.Pooling import PoolingLayer
from NN.Layers.FullyConnected import FullyConnectedLayer
from NN.Layers.Softmax import SoftmaxLayer
from NN.CifarLoader import load_data

def ReLU(z): return T.maximum(0.0, z)

params = {
    "layers": [4,10],
    "batchSize": 100,
    "maxEpochs": 5,
    "inputShape": [3,32,32],
    "output": 10,
    "data_split":[0.5, 0.75, 1],
    "learningRate": [0,2],
    "l2": [0,2]
}

def convolution(inpt, this):
    if len(inpt) != 3:
        return None
    this["filters"] = int(this["filters"])
    this["size"] = int(this["size"])
    image_shape = (params["batchSize"], inpt[0], inpt[1], inpt[2])
    filter_shape = (this["filters"], inpt[0], this["size"], this["size"])
    return ConvolutionalLayer(image_shape, filter_shape, activation_fn=ReLU)

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
    return FullyConnectedLayer(inpt, this["neurons"], this["dropout"], activation_fn=ReLU)

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
        "dropout": [0, 0.5],
        "fn": fullyconnected
    },
    "softmax":{
        "neurons": [params["output"],params["output"]],
        "dropout": [0, 0.5],
        "fn": softmax
    }
}

ga_params = {
    "maxGenerations": 50,
    "threshold": 0.99,
    "population": 20,
    "crossover": 0.2,
    "mutation": 0.1
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
    # randomise number of layers
    numLayers = random.randint(params["layers"][0],params["layers"][1])
    layers = []

    # start with a convolutional layer
    inpt = params["inputShape"]
    layer = generateLayer("convolution")
    layers.append(layer)

    # generate and append other layers
    for i in range(numLayers):
        k = list(definitions.keys())
        k.remove("softmax")
        typ = random.choice(k)
        layer = generateLayer(typ)
        layers.append(layer)

    # generate and append the output layer
    layer = generateLayer("softmax")
    layers.append(layer)

    # generate network parameters
    lr = random.uniform(params["learningRate"][0],params["learningRate"][1])
    lmbda = random.uniform(params["l2"][0],params["l2"][1])

    p = {
        "learningRate": lr,
        "l2": lmbda
    }

    # return individual
    return [layers, p]

def crossover(one, two):
    pointOne = random.randint(1,len(one[0])-2)
    pointTwo = random.randint(1,len(two[0])-2)
    resOne = [one[0][:pointOne] + two[0][pointTwo:], one[1]]
    resTwo = [two[0][:pointTwo] + one[0][pointOne:], two[1]]

    return resOne, resTwo

def mutate(ind):
    index = random.randint(0,len(ind[0])-1)
    key = "fn"
    forbidden = ["fn", "type"]
    while key in forbidden:
        key = random.choice(list(ind[0][index].keys()))

    typ = ind[0][index]["type"]
    new_value = int(random.uniform(definitions[typ][key][0],definitions[typ][key][1]))
    ind[0][index][key] = new_value
    return ind

def getFitness(fitness):
    if fitness is not None:
        return fitness[1][-1]
    else:
        return -1

def evaluate(ind):
    try:
        layers = ind[0]
        par = ind[1]
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

        fitness = [[-1],[-1]]
        fitness = net.train(training_data, params["maxEpochs"], params["batchSize"], par["learningRate"],
            validation_data, test_data, lmbda=par["l2"])
        return fitness
    except Exception as e:
        print("error in evaluation")
        print(e)
        return [[-1],[-1]]

def evaluateBest(ind):
    print(ind)
    # try:
    #     layers = ind[0]
    #     par = ind[1]
    #     dfns = []
    #     for i in range(len(layers)):
    #         inpt = None
    #         if len(dfns) == 0:
    #             inpt = params["inputShape"]
    #         else:
    #             inpt = dfns[-1].get_output_shape()
    #         dfn = defineLayer(inpt, layers[i])
    #         if dfn != None:
    #             dfns.append(dfn)
    #     net = NNet(dfns, params["batchSize"])
    #
    #     fitness = [[-1],[-1]]
    #     fitness = net.train(training_data, 100, params["batchSize"], par["learningRate"],
    #         validation_data, test_data, lmbda=par["l2"])
    #     print(fitness)
    #     return fitness
    # except Exception as e:
    #     print("error in evaluation")
    #     print(e)
    #     return [[-1],[-1]]

def processData(generations):
    print(generations[-1][0]["fitness"][1])
    print(generations[-1][0]["individual"])
    # generation = generations[0:13]
    # for index,generation in enumerate(generations):
    #     print(index)
    #     if index > 50:
    #         break
    #     for ind in generation:
    #         if ind["fitness"] is not None:
    #             print(ind["fitness"][1][-1]),
    #     print("")
    # print(len(generations))
    # # print(len(generations))
    # # for generation in generations:
    # #     print(len(generation))



# CONFIG
# try: theano.config.device = 'gpu'
# except: pass
theano.config.floatX = 'float32'
numpy.random.seed(0)
training_data, validation_data, test_data = load_data()

fns = {
    "generate": generate,
    "evaluate": evaluate,
    "evaluateBest": evaluateBest,
    "mutate": mutate,
    "crossover": crossover,
    "getFitness": getFitness,
    "processData": processData
}
dump_file = "dump.obj"
if len(sys.argv) > 1:
    if sys.argv[1] == "resume" and len(sys.argv) == 3:
        dump_file = sys.argv[2]
    elif len(sys.argv) == 2:
        dump_file = sys.argv[1]

ga = GAlg(ga_params, fns, dump_file)
# ga.evaluateBest()
# ga.keepFirst(13)
ga.processData()
# ga.run()
