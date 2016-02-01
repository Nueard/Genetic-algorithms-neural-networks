import random
import pickle
import os.path

class GAlg(object):
    def __init__(self, params, fns, dump_file):
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
                self.params = data["params"]
                self.generation = data["generation"]
                self.fns = data["fns"]
        else:
            self.dump_file = dump_file
            self.params = params
            self.params["generations"] = 0
            self.fns = fns
            self.generation = []

    def run(self):
        self.populate()
        for i in range(0, self.params["maxGenerations"]):
            self.evaluate()
            self.sort()
            print ("best individual in generation " + str(self.params["generations"]) + " has fitness of " + str(self.generation[0]["fitness"]))
            self.drop()
            self.crossover()
            self.populate()
            self.params["generations"] += 1
            self.saveProgress()
            if self.generation[0]["fitness"] >= self.params["threshold"]:
                print ("target threshold reached in " + str(i) + " generations")
                print ("best fitness " + str(self.generation[0]["fitness"]))
                print ("best individual \n" + str(self.generation[0]["individual"]))
                break

    def evaluate(self):
        print("Evaluating generation")
        for i in range(0,self.params["population"]):
            try:
                fitness = self.fns["evaluate"](self.generation[i]["individual"])
                self.generation[i]["fitness"] = fitness
            except:
                print("Exception while evaluating individual, skipping")

    def sort(self):
        def getKey(item):
            return item["fitness"]

        self.generation = sorted(self.generation, key=getKey, reverse=True)

    def drop(self):
        keep = int(int(self.params["population"] * 2 * self.params["crossover"]) / 2)
        self.generation = self.generation[:keep]

    def crossover(self):
        print("Mating top " + str(self.params["crossover"]*100) + " % of the population")
        for i in range(0, len(self.generation)-1):
            try:
                one, two = self.fns["crossover"](self.generation[i]["individual"],self.generation[i+1]["individual"])
                individual_one = {
                    "individual": one,
                    "fitness": 0
                }
                individual_two = {
                    "individual": two,
                    "fitness":0
                }
                individual_one = self.mutate(individual_one)
                individual_two = self.mutate(individual_two)
                self.generation.append(individual_one)
                self.generation.append(individual_two)
                i+=1
            except:
                print("Exception while mating individuals, skipping")

    def mutate(self):
        print("Trying to mutate")
        for i in range(0, len(self.generation)):
            rng = random.random()
            if rng < self.params["mutation"]:
                try:
                    self.generation[i]["individual"] = self.fns["mutate"](self.generation[i]["individual"])
                except:
                    print("Exception while mutating individual, skipping")

    def populate(self):
        print("Creating a new generation with " + str(self.params["population"]) + " individuals")
        for i in range(len(self.generation), self.params["population"]):
            individual = {}
            individual["individual"] = self.fns["generate"]()
            individual["fitness"] = 0
            self.generation.append(individual)

    def getBest(self):
        return self.generation[0]["individual"], self.generation[0]["fitness"]

    def saveProgress(self):
        data = {
            "fns": self.fns,
            "generation": self.generation,
            "params": self.params
        }
        with open(self.dump_file, 'wb') as handle:
            pickle.dump(data, handle)
