import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.special import expit


class Fitness:
    def __init__(self, alloc, n_succ, throughput):
        self.alloc = alloc
        self.n_succ = n_succ
        self.fitness = 0.0
        self.throughput = throughput

    def allocFitness(self):
        if self.fitness == 0:
            self.fitness = self.n_succ + np.finfo(float).eps + np.tanh(self.throughput / 100)
        return self.fitness


def createAlloc(n_cluster, power_options):
    # alloc = list(np.random.choice(list(power_options), n_cluster))
    alloc = list(np.zeros(n_cluster, dtype=np.int32))
    return alloc


def initialPopulation(popSize, n_cluster, power_options):
    population = []

    for i in range(popSize):
        population.append(createAlloc(n_cluster, power_options))
    return population


def rankAllocs(population, n_succ, throughput):
    fitnessResults = {}
    for i in range(len(population)):
        fitnessResults[i] = Fitness(population[i], n_succ[i], throughput[i]).allocFitness()
    result = sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)
    return result


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(eliteSize):
        selectionResults.append(popRanked[i][0])
    for _ in range(len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene):
        child.append(parent1[i])

    for i in range(startGene, endGene):
        child.append(parent2[i])

    for i in range(endGene, len(parent1)):
        child.append(parent1[i])

    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(eliteSize):
        children.append(matingpool[i])

    for i in range(length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate, power_options):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            individual[swapped] = np.random.choice(list(power_options), 1)[0]
    return individual


def mutatePopulation(population, mutationRate, power_options, eliteSize):
    mutatedPop = []

    for ind in range(eliteSize):
        mutatedPop.append(population[ind])

    for ind in range(eliteSize, len(population)):
        mutatedInd = mutate(population[ind], mutationRate, power_options)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate, n_succ, throughput, power_options):
    popRanked = rankAllocs(currentGen, n_succ, throughput)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate, power_options, eliteSize)
    return nextGeneration
