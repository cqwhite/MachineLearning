import numpy
import copy
import pprint


def normAndScale(fileName, places):
    inFile = open(fileName, "r")
    data = []
    names = ["sepal length", "sepal width", "petal length", "petal width", "target"]
    for line in inFile:
        if line[-1] == "\n":
            line = line[:-1]
        lineList = line.split(",")
        for idx in range(0, 4):
            lineList[idx] = float(lineList[idx])
        data.append(lineList)
    featureMeansList = []
    featuresMinList = []
    featuresMaxList = []
    for col in range(4):
        featureTotal = 0
        featuresMinList.append(data[col][0])
        featuresMaxList.append(data[col][0])
        for row in range(len(data)):
            featureTotal += data[row][col]
            if data[row][col] < featuresMinList[col]:
                featuresMinList[col] = data[row][col]
            if data[row][col] > featuresMaxList[col]:
                featuresMaxList[col] = data[row][col]
        featureMeansList.append(featureTotal / len(data))
    if places == -1:
        places = 16
    for row in range(len(data)):
        for col in range(4):
            data[row][col] = round(
                (data[row][col] - featureMeansList[col])
                / (featuresMaxList[col] - featuresMinList[col]),
                places,
            )
    featuresData = copy.deepcopy(data)
    for row in range(len(featuresData)):
        featuresData[row] = featuresData[row][:4]
    return data, featuresData[:150], len(featuresData)


data, featuresData, m = normAndScale("rawIris.txt", -1)
for lst in data:
    print(lst)
input("Normed and scaled Iris Data with classification - Press enter to continue . . .")
for lst in featuresData:
    print(lst)
input("Normed and scaled Iris Data - Press enter to continue . . .")

# Calc sigma method 1
X = numpy.array(featuresData)  # Turn into numpy array for easy manipulation
sigma = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
for vec in X:
    x = vec[:, numpy.newaxis]
    sigma = sigma + (x.transpose() * x)
sigma = sigma / m
print("Method 1 Calculation of Sigma\n", sigma)

# Calc sigma method 2 THIS IS PROBABLY THE METHOD I WOULD CHOOSE (MATRIX MULT WITH @)
sigma = (X.transpose() @ X) / m
print("Method 2 Calculation of Sigma\n", sigma)
input("Press enter to continue . . .")

U, S, V = numpy.linalg.svd(sigma)
print("\nU Matrix from svd\n", U)
print("\nS Matrix from svd", S)
Ureduce = U[:, 0:2]
variance = sum(S[0:1]) / sum(S)
k = S[0] / sum(S)
print(
    " Variance retained for k=1 is ", k * 100, "cumulative", round((variance * 100), 2)
)
variance = sum(S[0:2]) / sum(S)
k = S[1] / sum(S)
print(
    " Variance retained for k=2 is ", k * 100, "cumulative", round((variance * 100), 2)
)
variance = sum(S[0:3]) / sum(S)
k = S[2] / sum(S)
print(
    " Variance retained for k=3 is ", k * 100, "cumulative", round((variance * 100), 2)
)
variance = sum(S[0:4]) / sum(S)
k = S[3] / sum(S)
print(
    " Variance retained for k=4 is ", k * 100, "cumulative", round((variance * 100), 2)
)
print("\nV Matrix from svd\n", V, "\n\n")
input("Press enter to continue . . .")

Z = X @ Ureduce
print("The reduced feature X matrix (Z) is:\n", Z)
input("Press enter to continue . . .")
print("Reconstituting X from Z")
Xapprox = Z @ Ureduce.transpose()
print(Xapprox)
input("Press enter to continue . . .")

# Hmmm - how much diff between original and reconstituted X?
print("The difference between the original X and reconstituted X!")
diff = numpy.subtract(Xapprox, X)
for lst in diff:
    for item in lst:
        print(str(round(item, 3)).rjust(7), end="")
    print()

# RECAP OF COURSERA PCA LECTURE WITH DIMENSIONS SPELLED OUT
# Sigma = (1/m) * X' * X; % compute the covariance matrix (4, 150) (150, 4) => (4,4)
sigma = (X.transpose() @ X) / m
# [U,S,V] = svd(Sigma);   % compute all the projected dimensions using singular value decomposition on the covariance matrix (4,4)
U, S, V = numpy.linalg.svd(sigma)
# Ureduce = U(:,1:k);     % take the first k dimensions (4,2)
Ureduce = U[:, 0:2]
# Z = X * Ureduce;        % compute the projected data points (150,4) (4,2) => (150,2)
Z = X @ Ureduce
# TO GET Xapprox (the 'recreation' of the original X):
Xapprox = Z @ Ureduce.transpose()
