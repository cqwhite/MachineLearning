"""
Connor White & David Chalifloux

"""

import random
import matplotlib.pyplot as plt
import time
import sys

dataMatrix = [
    [0, 241, 162, 351, 183],
    [241, 0, 202, 186, 97],
    [162, 202, 0, 216, 106],
    [351, 186, 216, 0, 186],
    [183, 97, 106, 187, 0]
]


def genPop(numIndividuals, dataMatrix):
    indexMatrix = []
    for i in range(len(dataMatrix)-1):
        indexMatrix.append(i+1)
    
    population = []
    for i in range(numIndividuals):
        route = random.sample(indexMatrix, 4)
        population.append(route)
    return population


def fitness(chromo, dataMatrix):
    chromoList = [0, *chromo, 0]
    fitness = 0
    for i in range(len(chromoList)-1):
        x = chromoList[i]
        y = chromoList[i+1]
        fitness+= dataMatrix[x][y]

    return fitness


def naryTournament(pop, fitnessList, n=2):
    maxFit = sys.maxsize
    for rep in range(n):
        index = random.randrange(0, len(pop))
        if fitnessList[index] < maxFit:
            maxFit = fitnessList[index]
            mostFit = pop[index]
    return mostFit


def main(popSize, mutRate, crossRate, maxGens):
    pop = genPop(popSize, dataMatrix)
    maxFitList = []
    avgFitList = []
    for i in range(maxGens):
        fitnessList = []
        maxFit = sys.maxsize
        for individual in pop:
            fitnessList.append(fitness(individual, dataMatrix))
        maxFit = max(fitnessList)
        mostFitChromo = pop[fitnessList.index(maxFit)]
        maxFitList.append(maxFit)
        avgFit = sum(fitnessList)/len(fitnessList)
        avgFitList.append(avgFit)
        print("gen=", i, "maxfit=", maxFit, "avgfit=",
              avgFit, "bestChromo=", mostFitChromo)
        nextGeneration = []
        for reps in range(len(pop)//2):
            p1 = naryTournament(pop, fitnessList)
            p2 = naryTournament(pop, fitnessList)
            if random.random() <= crossRate:
                pass
            if random.random() <= mutRate:
                pointIndex = random.randrange(0,len(p1))
                secondPointIndex = random.randrange(0, len(p1))
                while (secondPointIndex == pointIndex):
                    secondPointIndex = random.randrange(0, len(p1))

                p1[pointIndex] = secondPointIndex
                p1[secondPointIndex] = pointIndex
            nextGeneration.append(o1)
            nextGeneration.append(o2)
            pop = nextGeneration[:]

    return "BestChromo=", mostFitChromo
    # generate population with random chromosome values
    # calculate fitnesses
    # tournament selection for reproduction
    # apply crossover and mutation



main(popSize=10, mutRate=.01, crossRate=.6, maxGens=10)
