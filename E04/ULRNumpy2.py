# Connor White
# Code modified from    https://data36.com/linear-regression-in-python-numpy-polyfit/
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

#import houseingData


def readAndSetUpTrainingValuesCSV(fileName):
    inFile = open(fileName, "r")
    xList = []
    yList = []

    csvFile = csv.DictReader(inFile)
    for line in csvFile:
        if line["city"] == "North Bend":
            xList.append(float(line["sqft_living"]))
            yList.append(float(line["price"]))

    return xList, yList


# Set up data
xList, yList = readAndSetUpTrainingValuesCSV(
    "E04/univariate_linear_regression.csv")
housingData = {'cost': xList, "sqr_ft": yList}
students = {'hours': [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43, 2, 39, 15, 44, 29, 41, 15, 24, 50],
            'test_results': [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73, 0, 62, 37, 74, 40, 90, 42, 58, 100]}
testStudents = {'hours': [1, 2, 3, 4, 5], 'test_results': [1, 2, 3, 4, 5]}
student_data = pd.DataFrame(data=housingData)
x = student_data.cost
y = student_data.sqr_ft

# Plot points
#plt.scatter(x, y)
# plt.show()

# Apply linear regression model (polynomial of degree 1) to data and show theta1 and theta0
degree = 3
model = np.polyfit(x, y, degree)
if degree == 1:
    print("Theta1 and Theta0", model)
else:
    print("Thetas . . .", model)

# Predict a new test score from hours studied
predict = np.poly1d(model)
hours_studied = 20
print("Predicted test score for", hours_studied,
      "hours studied:", predict(hours_studied))

# Show the r2 score (goodness of fit)
print("Goodness of fit (r2 score):", r2_score(y, predict(x)))

# Plot the points now with the regression line
x_lin_reg = range(int(min(x)), int(max(x)))
y_lin_reg = predict(x_lin_reg)
plt.scatter(x, y)
plt.plot(x_lin_reg, y_lin_reg, c='r')
plt.show()
