# Connor White & David Chalifloux

# From https://datascience.stackexchange.com/questions/26640/how-to-check-for-overfitting-with-svm-and-iris-data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from statistics import mean


iris = load_iris()
X = iris.data[:, :4]
y = iris.target

cList = [1, .1, .001, 100]
kernalList = ["linear", "rbf", "poly"]
degreeList = [2, 3, 4]

bestSVM = [0, "", "", ""]


def svmFunction(cParam, kernalParam, degreeParam=3):
    trainingAccuracyList = []
    testingAccuracyList = []

    for i in range(30):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        svm_model = svm.SVC(kernel=kernalParam, degree=degreeParam,
                            C=cParam, gamma='auto', probability=True)
        svm_model.fit(X_train, y_train)

        predictions = svm_model.predict(X_train)
        # print("Training =", accuracy_score(predictions, y_train))
        trainingAccuracyList.append(accuracy_score(predictions, y_train))

        predictions = svm_model.predict(X_test)
        # print("Testing  =", accuracy_score(predictions, y_test))
        testingAccuracyList.append(accuracy_score(predictions, y_test))

    trainingAccuracyMean = mean(trainingAccuracyList)
    testingAccuracyMean = mean(testingAccuracyList)
    return trainingAccuracyMean, testingAccuracyMean


for c in cList:
    for kernal in kernalList:
        if kernal == "poly":
            for degree in degreeList:
                means = svmFunction(c, kernal, degree)
                print("C =", c, "Kernal =", kernal, "Degree =", degree)
                print("Training", means[0], ", Testing", means[1])

                accuracyAverage = (means[0] + means[1])/2
                if (accuracyAverage > bestSVM[0]):
                    bestSVM[0] = accuracyAverage
                    bestSVM[1] = c
                    bestSVM[2] = kernal
                    bestSVM[3] = degree

        else:
            means = svmFunction(c, kernal)
            print("C =", c, "Kernal =", kernal)
            print("Training", means[0], ", Testing", means[1])

            accuracyAverage = (means[0] + means[1])/2
            if (accuracyAverage > bestSVM[0]):
                bestSVM[0] = accuracyAverage
                bestSVM[1] = c
                bestSVM[2] = kernal
                bestSVM[3] = ""

#THEN FIND THE BEST ONE##########
print("Best SVM", bestSVM)


# scores = cross_val_score(svm_model, iris.data, iris.target, cv=15)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

