import random
import sys
import pickle
import os.path
import matplotlib.pyplot as plt
import numpy

class GAlg(object):
    def __init__(self, params=None, fns=None, dump_file=None):
        """ Takes an dictionary of parameters 'params' and a dictionary of functions
            for evaluation, mutation, crossover and population (a.k.a. generation)

            'params':
                maxGenerations, int         : the maximum number of generations the algorithm will run
                population, int             : the number of individuals in a generation
                crossover, float (0 .. 1)   : top % to crossover e.g. 0.3 -> 30%
                mutation, float (0 .. 1)    : mutation % chance e.g. 0.5 -> 50%
                debug, bool                 : flag to turn debug info on
                threshold, float (0 .. 1)   : stop algorith if fitness is above

            'fns':
                evaluate, function          : evaluates the individual returning their fitness
                crossover, function         : returns the result of crossing two individuals
                muatate, function           : mutates an individual
                generate, function          : generates an individual
        """
        if os.path.isfile(dump_file):
            print("Found existing file dump, loading it now")
            with open(dump_file, 'rb') as handle:
                data = pickle.load(handle)
                self.dump_file = dump_file
                self.params = params
                self.params["generations"] = len(data["previousGenerations"])
                self.generation = data["generation"]
                self.previousGenerations = data["previousGenerations"]
                self.fns = fns
        else:
            self.dump_file = dump_file
            self.params = params
            self.params["generations"] = 0
            self.fns = fns
            self.generation = []
            self.previousGenerations = []

    def run(self):
        self.populate()
        for i in range(0, self.params["maxGenerations"]):
            try:
                self.previousGenerations.append(self.generation)
                self.evaluate()
                self.sort()
                print ("best individual in generation " + str(self.params["generations"]) + " has fitness of " + str(self.fns["getFitness"](self.generation[0]["fitness"])))
                self.saveProgress()
                self.drop()
                self.crossover()
                self.populate()
                self.params["generations"] += 1
                if self.fns["getFitness"](self.generation[0]["fitness"]) >= self.params["threshold"]:
                    print ("target threshold reached in " + str(i) + " generations")
                    print ("best fitness " + str(self.generation[0]["fitness"]))
                    print ("best individual \n" + str(self.generation[0]["individual"]))
                    break
            except KeyboardInterrupt:
                print("Interrupted")
                sys.exit()

    def evaluate(self):
        print("Evaluating generation "+str(len(self.previousGenerations)))
        for i in range(0,len(self.generation)):
            if len(self.generation) <= i:
                break
            if self.generation[i]["fitness"] is None:
                fitness = self.fns["evaluate"](self.generation[i]["individual"])
                self.generation[i]["fitness"] = fitness
                if numpy.max(fitness) == -1:
                    del self.generation[i]
                    i-=1
                else:
                    print("individual fitness " + str(self.fns['getFitness'](self.generation[i]["fitness"])))

    def sort(self):
        def getKey(item):
            return self.fns["getFitness"](item["fitness"])

        self.generation = sorted(self.generation, key=getKey, reverse=True)

    def drop(self):
        keep = int(int(self.params["population"] * 2 * self.params["crossover"]) / 2)
        self.generation = self.generation[:keep]

    def crossover(self):
        print("Mating top " + str(self.params["crossover"]*100) + " % of the population")
        for i in range(0, len(self.generation)-1):
            # try:
            one, two = self.fns["crossover"](self.generation[i]["individual"],self.generation[i+1]["individual"])
            individual_one = {
                "individual": self.mutate(one),
                "fitness": None
            }
            individual_two = {
                "individual": self.mutate(two),
                "fitness": None
            }
            self.generation.append(individual_one)
            self.generation.append(individual_two)
            i+=1
            # except KeyboardInterrupt:
            #     print("Interrupted")
            #     sys.exit()
            # except Exception as e:
            #     print("Error while mating")
            #     print(e)

    def mutate(self, ind):
        # print("Trying to mutate")
        rng = random.random()
        if rng < self.params["mutation"]:
            # try:
            ind = self.fns["mutate"](ind)
            # except KeyboardInterrupt:
            #     print("Interrupted")
            #     sys.exit()
            # except Exception as e:
            #     print("Error while mutating")
            #     print(e)
        return ind

    def populate(self):
        print("Creating a new generation with " + str(self.params["population"]) + " individuals")
        for i in range(len(self.generation), self.params["population"]):
            individual = {}
            individual["individual"] = self.fns["generate"]()
            individual["fitness"] = None
            self.generation.append(individual)

    def getBest(self):
        return self.generation[0]["individual"], self.generation[0]["fitness"]

    def saveProgress(self):
        data = {
            "fns": self.fns,
            "generation": self.generation,
            "params": self.params,
            "previousGenerations": self.previousGenerations
        }
        with open(self.dump_file, 'wb') as handle:
            pickle.dump(data, handle)

    def showBest(self):
        print(self.generation[0]["individual"])
        print(self.generation[0]["fitness"])

    def keepFirst(self, index):
        self.previousGenerations = self.previousGenerations[:index]
        self.saveProgress()

    def evaluateBest(self):
        self.fns["evaluateBest"](self.previousGenerations[-1][0])

    def processData(self):
        self.fns["processData"](self.previousGenerations)
