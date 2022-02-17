import time
import matplotlib.pyplot as plt

#Sample data
xList = [12, 21, 3, 14, 11, 10, 43, 32, 21]
yList = [14, 18, 8, 16, 8, 17, 39, 25, 12]

#Scatter plot and BFL Setup
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

#Show BFL and JCost Plots as Data Changes (simulated by a loop for this example)
for q in range(len(xList)):
    plt.pause(.05)
    lines[0].remove()
    # pair of x's, pair of y's, two points for line
    lines = BFLplot.plot([minX, maxX], [0, q*3])
    JcostPlotList.append(q*5)
    JcostPlotItersList.append(q)
    JcostPlot.scatter(JcostPlotItersList, JcostPlotList)
    #plt.pause(1.0)
#plt.show()
