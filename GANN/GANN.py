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

class GANN(object):
    def __init__(self, parameters, load_data, dump_file, processData):
        params = parameters["nn"]["params"]
        definitions = parameters["nn"]["definitions"]
        ga_params = parameters["ga"]
        definitions["softmax"]["neurons"] = [params["output"],params["output"]]

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

        definitions["convolution"]["fn"] = convolution
        definitions["pooling"]["fn"] = pooling
        definitions["fullyconnected"]["fn"] = fullyconnected
        definitions["softmax"]["fn"] = softmax

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
                return numpy.max(fitness[1])
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
                print("Individual fitness " + str(numpy.average(fitness[1])))
                return fitness
            except Exception as e:
                print("error in evaluation")
                print(e)
                return [[-1],[-1]]

        def evaluateBest(ind):
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
                fitness = net.train(training_data, 100, params["batchSize"], par["learningRate"],
                    validation_data, test_data, lmbda=par["l2"])
                print("Individual fitness " + str(numpy.average(fitness[1])))
                print(fitness)
                return fitness
            except Exception as e:
                print("error in evaluation")
                print(e)
                return [[-1],[-1]]

        fns = {
            "generate": generate,
            "evaluate": evaluate,
            "evaluateBest": evaluateBest,
            "mutate": mutate,
            "crossover": crossover,
            "getFitness": getFitness,
            "processData": processData
        }

        try: theano.config.device = 'gpu'
        except: pass
        theano.config.floatX = 'float32'
        numpy.random.seed(0)
        training_data, validation_data, test_data = load_data(data_split=params["data_split"])

        self.dump_file = dump_file
        if len(sys.argv) > 1:
            if sys.argv[1] == "resume" and len(sys.argv) == 3:
                dump_file = sys.argv[2]

        self.ga = GAlg(ga_params, fns, dump_file)

    def run(self):
        self.ga.run()
        self.ga.processData()