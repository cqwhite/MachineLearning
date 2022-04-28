import random
import matplotlib.pyplot as plt
import time
import sys

pows2 = [1, 2, 4, 8, 16, 32, 64, 128, 256,
         512, 1024, 2048, 4096, 8192, 16384, 32768]


def genPop(numIndividuals, lenChromo):
    population = []
    for i in range(numIndividuals):
        bitList = [['0', '1'][random.randint(0, 1)] for i in range(lenChromo)]
        bitString = ''.join(bitList)
        population.append(bitString)
    return population


def fitness(chromo, geneBoundaries):
    chromoX = chromo[0:geneBoundaries[0]]
    totalX = int(chromoX, 2)
    chromoY = chromo[geneBoundaries[0]:]
    totalY = int(chromoY, 2)
    return totalX-totalY


def naryTournament(pop, fitnessList, n=2):
    maxFit = sys.maxsize*-1
    for rep in range(n):
        index = random.randrange(0, len(pop))
        if fitnessList[index] > maxFit:
            maxFit = fitnessList[index]
            mostFit = pop[index]
    return mostFit


def main(popSize, chromoSize, geneBoundaries, mutRate, crossRate, maxGens, convergePct):
    pop = genPop(popSize, chromoSize)
    maxFitList = []
    avgFitList = []
    for i in range(maxGens):
        fitnessList = []
        maxFit = sys.maxsize*-1
        for individual in pop:
            fitnessList.append(fitness(individual, geneBoundaries))
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
                coop = random.randrange(0, len(p1))
                o1 = p1[:coop]+p2[coop:]
                o2 = p2[:coop]+p1[coop:]
            else:
                o1 = p1
                o2 = p2
            if random.random() <= mutRate:
                pointIndex = random.randrange(0, len(p1))
                if o1[pointIndex] == "1":
                    o1 = o1[:pointIndex]+"0"+o1[pointIndex+1:]
                else:
                    o1 = o1[:pointIndex]+"1"+o1[pointIndex+1:]
            if random.random() <= mutRate:
                pointIndex = random.randrange(0, len(p1))
                if o2[pointIndex] == "1":
                    o2 = o2[:pointIndex]+"0"+o2[pointIndex+1:]
                else:
                    o2 = o2[:pointIndex]+"1"+o2[pointIndex+1:]
            nextGeneration.append(o1)
            nextGeneration.append(o2)
            pop = nextGeneration[:]

    pass
    # generate population with random chromosome values
    # calculate fitnesses
    # tournament selection for reproduction
    # apply crossover and mutation


main(popSize=500, chromoSize=32, geneBoundaries=[
     16], mutRate=.01, crossRate=.6, maxGens=1000, convergePct=1.0)
