# Starter Code for multivariate regression with train and test
# Art White, February, 2022
# Connor White, David Chalifoux
import numpy as np
import pandas as pd
import random
from scipy import rand


def hypothesis(thetas, Xrow):
    dependentY = 0
    for idx in range(len(thetas)):
        dependentY += thetas[idx]*Xrow[idx]
    return round(dependentY, 0)


def normalEquation(XsTitles, housingDF):
    print(XsTitles)
    XsDF = housingDF[XsTitles]
    YDF = housingDF['price']
    # Convert dataframes to numpy ndarray(matrix) types
    Xs = XsDF.to_numpy()
    Y = YDF.to_numpy()
    # Add the 1's column to the Xs matrix (1 * the intercept values, right?)
    XsRows, XsCols = Xs.shape
    X0 = np.ones((XsRows, 1))
    Xs = np.hstack((X0, Xs))

    # Calc the Thetas via the normal equation
    thetas = (np.linalg.pinv(Xs.T @ Xs)) @ Xs.T @ Y
    print("Thetas:", thetas)

    # Now, generate differences from the predicted
    predictedM = (Xs @ thetas.T)
    diffs = abs(predictedM-Y)
    sumOfDiffs = diffs.sum()
    sumOfPrices = Y.sum()
    prediction = round(sumOfDiffs/sumOfPrices*100, 1)
    print("average price difference for training values:",
          str(prediction)+"%")
    return prediction


# Read data from file using pandas and create a dataframe
housingDF = pd.read_csv('E06/housing.csv')
northBendHousingDF = housingDF[housingDF['city'] == 'North Bend']
columnNames = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
               "waterfront", "view", "condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated"]


# Subdivide the data into features (Xs) and dependent variable (y) dataframes
#XsTitles = ['sqft_living', 'bedrooms', 'bathrooms', 'yr_built', 'waterfront']

#find the best fit for features
equationFlag = True
count = 0
while(equationFlag):
    XsTitles = []
    for i in range(6):
        randomIndex = random.randint(0, len(columnNames)-1)
        XsTitles.append(columnNames[randomIndex])
    x = normalEquation(XsTitles, housingDF)
    count = count + 1
    if(x < 30.9):
        equationFlag = True
        break
print("Iterations: " + str(count))
