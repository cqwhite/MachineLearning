# Starter Code for multivariate regression with train and test
# Art White, February, 2022
# Connor White & David Chalifoux
import numpy as np
import pandas as pd


def hypothesis(thetas, Xrow):
    dependentY = 0
    for idx in range(len(thetas)):
        dependentY += thetas[idx]*Xrow[idx]
    return round(dependentY, 0)


# Read data from file using pandas and create a dataframe
housingDF = pd.read_csv('housing.csv')
trainingDF = housingDF.sample(frac=.3).reset_index()
testingDF = housingDF.drop(trainingDF.index).reset_index()

# Subdivide the data into features (Xs) and dependent variable (y) dataframes
XsTitles = ['view', 'floors', 'bathrooms',
            'sqft_living', 'bedrooms', 'yr_built']
XsDF = trainingDF[XsTitles]
YDF = trainingDF['price']
XsTestingDF = testingDF[XsTitles]
YTestingDF = testingDF['price']
# Convert dataframes to numpy ndarray(matrix) types
Xs = XsDF.to_numpy()
Y = YDF.to_numpy()
XsTesting = XsTestingDF.to_numpy()
YTesting = YTestingDF.to_numpy()
# Add the 1's column to the Xs matrix (1 * the intercept values, right?)
XsRows, XsCols = Xs.shape
X0 = np.ones((XsRows, 1))
Xs = np.hstack((X0, Xs))

XsTestingRows, XsTestingCols = XsTesting.shape
X0Testing = np.ones((XsTestingRows, 1))
XsTesting = np.hstack((X0Testing, XsTesting))

# Calc the Thetas via the normal equation
thetas = (np.linalg.pinv(Xs.T @ Xs)) @ Xs.T @ Y

# Now, generate differences from the predicted
# predictedM = (Xs @ thetas.T)
# diffs = abs(predictedM-Y)
# sumOfDiffs = diffs.sum()
# sumOfPrices = Y.sum()
# print("average price difference for training values:",
#       str(round(sumOfDiffs/sumOfPrices*100, 1))+"%")
# OR
# Now, generate differences from the predicted (could be done iteratively)
totalDiffs = 0
totalPrices = 0
for row in range(len(Xs)):
    totalDiffs += abs(Y[row]-hypothesis(thetas, Xs[row]))
    totalPrices += Y[row]
print("average price difference for training values:",
      str(round(totalDiffs/totalPrices*100, 1))+"%")

# Generate differences from testing vs. hypothesis
totalDiffs = 0
totalPrices = 0
for row in range(len(XsTesting)):
    totalDiffs += abs(YTesting[row]-hypothesis(thetas, XsTesting[row]))
    totalPrices += YTesting[row]
print("average price difference for testing values:",
      str(round(totalDiffs/totalPrices*100, 1))+"%")
