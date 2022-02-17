# Connor White

import time
import matplotlib.pyplot as plt

""" The following code is meant only for finding a univariate
(one feature) linear regression line given a set of data"""
import time
global DEBUG
DEBUG = False

# Code the data reading function to produce the vectors
# xList (our single feature) and yList (our actual y value) corresponding
# to the parallel x value in xList when it was observed.  It should be
# noted that the same x value may result in a different observed
# value in our training data set.  We expect to read in m training
# pairs (x,y) from our file.
# Input: the file name of a file with x value comma y value per line
# Outputs: xList, yList (parallel with xList), m (the number of #training cases)


def livePlotGraph(xList, yList):
    # Scatter plot and BFL Setup
    BFLplot = plt.subplot(2, 1, 1)
    JcostPlot = plt.subplot(2, 1, 2)
    maxX = max(xList)
    minX = min(xList)
    maxY = max(yList)
    minY = min(yList)
    plt.ion()
    BFLplot.scatter(xList, yList, 20, color=(0, 0, 0), alpha=.3)
    BFLplot.set_title('Scatter Plot and Best-Fit Lines over Iterations')
    BFLplot.set_xlabel('x')
    BFLplot.set_ylabel('y')
    # pair of x's, pair of y's, two points for line
    lines = BFLplot.plot([minX, maxX], [0, 0])
    JcostPlot.set_title('JCost (Error) Plot')
    JcostPlot.set_xlabel('Iterations')
    JcostPlot.set_ylabel('Error')
    JcostPlotList = []
    JcostPlotItersList = []

    # Show BFL and JCost Plots as Data Changes (simulated by a loop for this example)
    for q in range(len(xList)):
        plt.pause(.01)
        lines[0].remove()
        # pair of x's, pair of y's, two points for line
        lines = BFLplot.plot([minX, maxX], [0, q*3])
        JcostPlotList.append(q*5)
        JcostPlotItersList.append(q)
        JcostPlot.scatter(JcostPlotItersList, JcostPlotList)
        # plt.pause(1.0)
    # plt.show()


def readAndSetUpTrainingValues(fileName):
    inFile = open(fileName, "r")
    xList = []
    yList = []
    m = 0
    for line in inFile:
        items = line.split(",")
        xList.append(float(items[0]))
        yList.append(float(items[1]))
        m += 1
    return xList, yList, m


def readAndSetUpTrainingValuesCSV(fileName):
    inFile = open(fileName, "r")
    xList = []
    yList = []
    m = 0
    for line in inFile:
        # if (m != 0):
        items = line.split(",")
        if (items[15] == "North Bend"):
            # print(items[15])
            xList.append(float(items[1]))
            yList.append(float(items[4]))
            m += 1
       # m = m - 1
    return xList, yList, m

# Code the hypothesis function
# Inputs: x value, theta0 and theta1
# Output: predicted value (predicted y)


def hOfX(x, theta0, theta1):
    return theta0+theta1*x

# Code the cost function.
# Not used directly for linear regression (the derivative
# is used to adjust #thetas in our gradient descent algorithm)
# but can be output as an indication that the training process
# is moving in the right direction, or used to terminate the
# gradient descent loop.
# Invokes hOfX
# Inputs: m, theta0, theta1, xList, yList
# Output: cost valuedef JCost(m,theta0,theta1,xList,yList):


def JCost(m, theta0, theta1, xList, yList):
    sumSquaredDiffs = 0
    for i in range(m):
        sumSquaredDiffs += (hOfX(xList[i], theta0, theta1)-yList[i])**2
    cost = 1/(2*m)*sumSquaredDiffs
    return cost

# Code the partial derivative of JCost (given in notes) for use
# in the gradient descent process
# Inputs: m, theta0, theta1, xList, yList
# Outputs: totalDiffs for Theta0 and Theta1


def JCostDerivForGradientDescent(m, theta0, theta1, xList, yList):
    totalDiffsForTheta0 = 0
    totalDiffsForTheta1 = 0
    for i in range(m):
        temp = hOfX(xList[i], theta0, theta1)-yList[i]
        totalDiffsForTheta0 += temp
        totalDiffsForTheta1 += temp*xList[i]
    return totalDiffsForTheta0, totalDiffsForTheta1

# Code the gradient descent process loop for convergence on
# the global minimum (for univariate linear regression should
# always be able to find a near minimum that exists for JCost
# if alpha is correctly chosen)
# Inputs: m, xList,yList,alpha,threshold,maxIters
# Outputs: theta0, theta1, (doc purposes: alpha, countIters, threshold)


def gradientDescent(m, xList, yList, alpha, threshold, maxIters):
    countIters = 0
    theta0 = 0
    theta1 = 0
    currentJCost = JCost(m, theta0, theta1, xList, yList)
    prevJCost = currentJCost+1
    if DEBUG:
        print(currentJCost)
    while countIters < maxIters and abs(prevJCost-currentJCost) > threshold:
        tDiffsTheta0, tDiffsTheta1 = JCostDerivForGradientDescent(
            m, theta0, theta1, xList, yList)
        theta0 = theta0 - alpha * ((1/m) * (tDiffsTheta0))
        theta1 = theta1 - alpha * ((1/m) * (tDiffsTheta1))
        prevJCost = currentJCost
        currentJCost = JCost(m, theta0, theta1, xList, yList)
        if currentJCost > prevJCost:
            print("JCost is increasing - why?")
            exit()
        if countIters % 10000 == 0 or countIters in range(30):
            print('Iters={:7d} prevJCost={:01.10f} currentJCost={:01.10f} diff={:01.10f}'.format(
                countIters, prevJCost, currentJCost, prevJCost-currentJCost))
        countIters += 1
        if DEBUG:
            print(currentJCost)
    # print(tDiffsTheta0,tDiffsTheta1,currentJCost)
    if DEBUG:
        print("Gradient Descent iterations", countIters)
    return theta0, theta1, alpha, countIters, threshold


def univariateLinearRegression(fileName, alpha, threshold, maxIters):
    # xList,yList,m=readAndSetUpTrainingValues("versicolorSepalWidthSepalLength.csv")
    if (fileName == "univariate_linear_regression.csv"):
        xList, yList, m = readAndSetUpTrainingValuesCSV(fileName)
    else:
        xList, yList, m = readAndSetUpTrainingValues(fileName)
    if DEBUG:
        print(xList)
    if DEBUG:
        print(yList)
    if DEBUG:
        print(m)
    start = time.time()
    theta0, theta1, alpha, iters, threshold = gradientDescent(
        m, xList, yList, alpha, threshold, maxIters)
    stop = time.time()
    print('Theta0= {:01.17f}   Theta1= {:01.17f}'.format(theta0, theta1))
    print("For alpha", format(alpha, '0.12f').rstrip('0'), "with", iters, "iterations", "and threshold",
          format(threshold, '.21f').rstrip('0'), "took", format(stop-start, '.7f'), "seconds")
    livePlotGraph(xList, yList)


# univariateLinearRegression("test_validity.txt",alpha=.1,threshold=0.1e-20,maxIters=1e10)
# univariateLinearRegression("North_Bend.txt",alpha=.1,threshold=0.1e-10,maxIters=1e5)

univariateLinearRegression("univariate_linear_regression.csv",
                           alpha=.00000000001, threshold=0.1e-10, maxIters=1e5)

#univariateLinearRegression("testData.txt", alpha=.00000000000001, threshold=0.1e-10, maxIters=1e5)
