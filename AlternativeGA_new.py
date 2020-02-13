import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
# from scipy.special import expit
import copy


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

def initialPopulation(popSize, n_cluster, fsb_pl, scaled_fsb_pl, OneDim_opt_pl, max_pl):
    population = []

    # initial population based on feasibility power
    pl_vec = np.asarray(fsb_pl.copy())
    population.append(list(pl_vec))
    dice_roll = np.random.rand(n_cluster)
    is_up = (dice_roll > 0.5)
    pl_vec[is_up] = pl_vec[is_up] + 1
    population.append(list(pl_vec))
    for i in range(int(popSize / 5) - 2):
        dice_roll = np.random.rand(n_cluster)
        is_up = (dice_roll > 2 / 3)
        is_down = (dice_roll < 1 / 3)
        pl_vec[is_up] = pl_vec[is_up] + 1
        pl_vec[is_down] = pl_vec[is_down] - 1
        indices_roundup = ((pl_vec - fsb_pl) < 0)
        indices_rounddown = ((pl_vec - max_pl) > 0)
        pl_vec[indices_roundup] = pl_vec[indices_roundup] + 1
        pl_vec[indices_rounddown] = pl_vec[indices_rounddown] - 1
        population.append(list(pl_vec))

    # initial population based on scaled feasibility power
    pl_vec = np.asarray(scaled_fsb_pl.copy())
    population.append(list(pl_vec))
    for i in range(int(popSize / 5) - 1):
        dice_roll = np.random.rand(n_cluster)
        is_up = (dice_roll > 2 / 3)
        is_down = (dice_roll < 1 / 3)
        pl_vec[is_up] = pl_vec[is_up] + 1
        pl_vec[is_down] = pl_vec[is_down] - 1
        indices_roundup = ((pl_vec - fsb_pl) < 0)
        indices_rounddown = ((pl_vec - max_pl) > 0)
        pl_vec[indices_roundup] = pl_vec[indices_roundup] + 1
        pl_vec[indices_rounddown] = pl_vec[indices_rounddown] - 1
        population.append(list(pl_vec))

    # initial population based on 1-d optimized power
    power_vec = np.asarray(OneDim_opt_pl.copy())
    population.append(list(power_vec))
    for i in range(int(popSize / 5) - 1):
        dice_roll = np.random.rand(n_cluster)
        is_up = (dice_roll > 2 / 3)
        is_down = (dice_roll < 1 / 3)
        pl_vec[is_up] = pl_vec[is_up] + 1
        pl_vec[is_down] = pl_vec[is_down] - 1
        indices_roundup = ((pl_vec - fsb_pl) < 0)
        indices_rounddown = ((pl_vec - max_pl) > 0)
        pl_vec[indices_roundup] = pl_vec[indices_roundup] + 1
        pl_vec[indices_rounddown] = pl_vec[indices_rounddown] - 1
        population.append(list(pl_vec))

    # initial population based on random power selection
    for i in range(int(popSize / 5) * 2):
        dice_roll = np.random.rand(n_cluster)
        power_vec = np.asarray(fsb_pl) + dice_roll * (np.asarray(max_pl) - np.asarray(fsb_pl))
        power_vec = power_vec.astype(np.int32)
        population.append(list(power_vec))

    return population


def rankAllocs(population, n_succ, throughput):
    fitnessResults = {}
    for i in range(len(population)):
        fitnessResults[i] = Fitness(population[i], n_succ[i], throughput[i]).allocFitness()
    result = sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)
    return result


def select_elites(popRanked, eliteSize, population):
    elites = []

    for i in range(eliteSize):
        elites.append(population[popRanked[i][0]])

    return elites


def select_mating(popRanked, eliteSize, population):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(eliteSize):
        selectionResults.append(popRanked[i][0])
    for _ in range(len(popRanked) - 2 * eliteSize):
        pick = 100 * random.random()
        for i in range(len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break

    matingpool = []
    for i in range(len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])

    return matingpool


def mating(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(eliteSize):
        children.append(matingpool[i])

    for i in range(length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


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


def mutate(population, mutationRate, fsb_pl, max_pl):
    n_clusters = len(population[0])

    mutated_pop = []
    for i in range(len(population)):
        is_mutated = (np.random.rand(n_clusters) < mutationRate)
        mutated_individual = np.asarray(population[i])
        dice_roll = np.random.rand(np.sum(is_mutated))
        mutated_individual[is_mutated] = fsb_pl[is_mutated] + dice_roll * (max_pl[is_mutated] - fsb_pl[is_mutated])
        mutated_pop.append(list(mutated_individual))

    return mutated_pop


def nextGeneration(currentGen, eliteSize, n_succ, throughput, n_cluster, mutationRate, fsb_pl, max_pl):
    popRanked = rankAllocs(currentGen, n_succ, throughput)
    elites = select_elites(popRanked, eliteSize, currentGen)

    nextGeneration = []

    for i in range(eliteSize):
        nextGeneration.append(elites[i])

    mating_pool = select_mating(popRanked, eliteSize, currentGen)
    children = mating(mating_pool, eliteSize)
    mutated_children = mutate(children, mutationRate, fsb_pl, max_pl)

    for child in mutated_children:
        nextGeneration.append(list(child))

    return nextGeneration
