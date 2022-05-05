"""
Connor White & David Chalifloux

"""

import random
import sys


def genPop(numIndividuals, dataMatrix):
    indexMatrix = []
    for i in range(len(dataMatrix) - 1):
        indexMatrix.append(i + 1)

    population = []
    for i in range(numIndividuals):
        route = random.sample(indexMatrix, 4)
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
                nextPopulation.append(mate1)
            else:
                nextPopulation.append(mate2)
    return nextPopulation


def main(popSize, mutRate, crossRate, maxGens):
    dataMatrix = [
        [0, 241, 162, 351, 183],
        [241, 0, 202, 186, 97],
        [162, 202, 0, 216, 106],
        [351, 186, 216, 0, 186],
        [183, 97, 106, 187, 0],
    ]

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
            "avgfit:",
            avgFit,
            "bestChromo:",
            mostFitChromo,
        )
        population = nextGeneration(
            population, fitnessList, mutRate, crossRate, dataMatrix
        )


main(popSize=10, mutRate=0.01, crossRate=0.6, maxGens=100)
