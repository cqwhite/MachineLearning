"""
Connor White & David Chalifoux

What preparations are needed to make the data useable?
Thanks to our implementation, nothing.

What decisions will need to be made regarding the GA configuration? In other words, what are the parameters that will affect overall GA performance?
Population size, mutation rate, crossover rate, and max generations will be most useful to us.
Because the length of each chromosome is quite long (30), my intuition tells me that a larger population size will be beneficial.

What configuration provided the best performance as a % of optimal tour?
popSize=50000, mutRate=0.1, crossRate=0.4, maxGens=115
Most reliably trended toward at least 95% fitness.

What appears to be your best tour?
My best run had a fitness percentage of 99.9%.

What is the number of possible tours that would have to be tested to find the brute force solution?
8.841762e+30 (29 factorial)

"""

import random
import sys
import pickle


def readData(fileName):
    if fileName[-4:] == ".pkl":
        print("Reading distance matrix from .pkl file")
        pklIn = open(fileName, "rb")
        distanceLOL = pickle.load(pklIn)
        pklIn.close()
        return distanceLOL
    else:
        print(
            "Expects a pickled file containing a list of lists for the distance matrix!"
        )


def genPop(numIndividuals, dataMatrix):
    indexMatrix = []
    for i in range(len(dataMatrix) - 1):
        indexMatrix.append(i + 1)

    population = []
    for i in range(numIndividuals):
        route = random.sample(indexMatrix, len(dataMatrix) - 1)
        population.append(route)
    return population


def fitness(chromo, dataMatrix):
    chromoList = [0, *chromo, 0]
    fitness = 0
    for i in range(len(chromoList) - 1):
        x = chromoList[i]
        y = chromoList[i + 1]
        fitness += dataMatrix[x][y]
    return fitness


def naryTournament(population, fitnessList, n=2):
    minFit = sys.maxsize
    for _ in range(n):
        index = random.randrange(0, len(population))
        if fitnessList[index] < minFit:
            minFit = fitnessList[index]
            mostFit = population[index]
    return mostFit


def mutation(child, mutRate=0.01):
    if random.random() <= mutRate:
        indices = random.sample(range(0, len(child)), k=2)
        temp1 = child[indices[0]]
        temp2 = child[indices[1]]
        child[indices[0]] = temp2
        child[indices[1]] = temp1
    return child


def nextGeneration(population, fitnessList, mutRate, crossRate, dataMatrix):
    nextPopulation = []
    while len(nextPopulation) < len(population):
        # Select two mates using tournament
        mate1 = naryTournament(population, fitnessList)
        mate2 = naryTournament(population, fitnessList)

        # Crossover based on crossRate
        if random.random() <= crossRate:
            # Randomly select segment from mate1
            segmentStart = random.randrange(0, len(mate1))
            segmentLength = 1
            remainingLength = len(mate1[segmentStart:])
            if remainingLength > 1:
                segmentLength = random.randrange(1, len(mate1[segmentStart:]))
            else:
                segmentLength = 1
            segment = mate1[segmentStart : segmentStart + segmentLength]

            # Create child
            child = [None] * len(mate1)
            child[segmentStart : segmentStart + segmentLength] = segment
            nextIndex = segmentStart + segmentLength
            while None in child:
                if nextIndex > len(child) - 1:
                    nextIndex = 0

                while mate2[nextIndex] in child:
                    nextIndex += 1
                    if nextIndex > len(child) - 1:
                        nextIndex = 0
                child[nextIndex] = mate2[nextIndex]
                nextIndex += 1
            child = mutation(child, mutRate)
            nextPopulation.append(child)
        else:
            # Perserve best fit mate
            if fitness(mate1, dataMatrix) < fitness(mate2, dataMatrix):
                child = mutation(mate1, mutRate)
                nextPopulation.append(child)
            else:
                child = mutation(mate2, mutRate)
                nextPopulation.append(child)
    return nextPopulation


def main(popSize, mutRate, crossRate, maxGens):
    dataMatrix = readData("E21/westernSahara29PickledMatrix.pkl")
    population = genPop(popSize, dataMatrix)
    minFitLists = []
    for i in range(maxGens):
        fitnessList = []
        for individual in population:
            fitnessList.append(fitness(individual, dataMatrix))
        minFit = min(fitnessList)
        mostFitChromo = population[fitnessList.index(minFit)]
        minFitLists.append(minFit)
        avgFit = sum(fitnessList) / len(fitnessList)
        print(
            "gen:",
            i,
            "minFit:",
            minFit,
            "percentage:",
            27603 / minFit,
        )
        population = nextGeneration(
            population, fitnessList, mutRate, crossRate, dataMatrix
        )


main(popSize=50000, mutRate=0.1, crossRate=0.4, maxGens=115)
