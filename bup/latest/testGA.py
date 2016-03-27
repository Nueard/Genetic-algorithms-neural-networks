from GA.GA import GAlg
import random

length = 100

def generate():
    result = []
    for i in range(0,length):
        bit = random.randint(0,1)
        result.append(bit)
    return result

def crossover(one, two):
    point = random.randint(1,length)
    resOne = one[:point] + two[point:]
    resTwo = one[point:] + two[:point]
    return resOne, resTwo

def mutate(ind):
    i = random.randint(0,length-1)
    if ind[i] == 1:
        ind[i] = 0
    else:
        ind[i] = 1
    return ind

def evaluate(ind):
    count = 0
    for i in range(0,length):
        if ind[i] == 1:
            count+=1

    return count/float(length)

params = {
    "maxGenerations": 100,
    "threshold": 1,
    "population": 100,
    "crossover": 0.25,
    "mutation": 0.01
}

fns = {
    "generate": generate,
    "evaluate": evaluate,
    "mutate": mutate,
    "crossover": crossover
}

ga = GAlg(params, fns)
ga.run()
