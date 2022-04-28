import random
import matplotlib.pyplot as plt
import time

pows2=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]

def genPop(numIndividuals, lenChromo):
    population=[]
    for i in range(numIndividuals):
        bitList = [['0','1'][random.randint(0,1)] for i in range(lenChromo)]
        bitString =''.join(bitList)
        population.append(bitString)
    return population

def fitness(chromo,pows2,geneBoundaries):
    pass
    return totalX-totalY

def naryTournament(pop,fitnessList,n=2):
    pass
    return P


def main(popSize,chromoSize,geneBoundaries,mutRate,crossRate,maxGens,convergePct):
    pass
    #generate population with random chromosome values
    #repeat maxGens times
        #calculate fitnesses
        #tournament selection for reproduction
        #apply crossover and mutation

main(popSize=500,chromoSize=32,geneBoundaries=[16],mutRate=.01,crossRate=.6,maxGens=100,convergePct=1.0)   
