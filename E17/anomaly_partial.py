import math

def readFile(infile, data):
    file1 = open(infile,"r")
    for i in range(6): #skip the header lines
        file1.readline()
    for line in file1:
        lineList = line.split(" ")
        if len(lineList) > 1:
            vals = [float(lineList[1]),float(lineList[2])]    
            data.append(vals)
            
def readY(infile, data):
    file1 = open(infile,"r")
    for i in range(6): #skip the header lines
        file1.readline()
    for line in file1:
        if line[0] == " ":
            data.append(int(line))

def main():
    #Get training and xval data
    xList = []
    xValList = []
    yValList = []
    readFile("X.txt", xList)
    readFile("Xval.txt", xValList)
    readY("yval.txt", yValList)
    #calc mu's for feature 1 and feature 2
    mu1 = 0
    mu2 = 0
    for elem in xList:
        mu1 += elem[0]
        mu2 += elem[1]
    mu1 /= len(xList)
    mu2 /= len(xList)
    #calc sigmas for feature 1 and feature 2
    sigma1 = 0
    sigma2 = 0
    for elem in xList:
        sigma1 += (elem[0] - mu1)**2
        sigma2 += (elem[1] - mu2)**2
    sigma1 /= len(xList)
    sigma2 /= len(xList)

    f1 = 0
    prevf1 = -1
    epsilon = .01
    numIters = 0
    while f1 >= prevf1:
        prevf1 = f1
        prevepsilon = epsilon
        epsilon = epsilon/2
        anomalyList = []
        for x in xValList:
            #mod this code to calc p1,p2 and p
            p1 = 0.0
            p2 = 0.0
            p = 0.0
            if p < epsilon:
                anomalyList.append(1)
            else:
                anomalyList.append(0)

        truePositives = 0
        falsePositives = 0
        falseNegatives = 0
        for i in range(len(anomalyList)):
            #your code here to count TP,FP and FN
            pass
        precision = truePositives/(truePositives + falsePositives)
        recall = truePositives/(truePositives + falseNegatives)
        f1 = 2 * (precision * recall)/(precision + recall)
        print("true positives =",truePositives,"   false positives =",falsePositives,"   false negatives =",falseNegatives)
        print("f1 = %.10f" % f1)
        print("epsilon = %.10f" % epsilon)
        print("------------------")
        numIters += 1

    print("\n------------------")
    print("Final f1 = %.10f" % prevf1)
    print("Final epsilon = %.10f" % prevepsilon)
    print("Num iters = ", numIters)
    print("------------------")

main()
