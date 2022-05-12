"""
Example of using a nuerel network for detecting an anomoly (true or false (0, 1))

RESULTS:
    50 Epochs:
    loss: 0.5907 - accuracy: 0.7227
    Train Accuracy: 0.723
    Test Accuracy: 0.746

David Chalifoux, Connor White, Quinn Partain, Micah Odell
"""
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import numpy as np


def MLP(activationOne, activationTwo, nodeCountOne, NodeCountTwo, testSize, epochsNumber):
    # load the dataset
    fileName = "./KratosProject/MP1/sat41589.txt"
    inFile = open(fileName, 'r')
    df = read_csv(inFile, header=None)
    # split into input and output columns
    X, y = df.values[:, 2:-1], df.values[:, -1]
    X = np.asarray(X).astype('float32')
    y = np.asarray(y).astype('float32')
    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testSize)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # determine the number of input features
    n_features = X_train.shape[1]
    print("n=", n_features)

    # define model
    model = Sequential()
    model.add(Dense(nodeCountOne, activation=activationOne,
              input_shape=(n_features,)))
    model.add(Dense(NodeCountTwo, activation=activationTwo))

    # compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(X_train, y_train, epochs=epochsNumber, verbose=2)

    # evaluate the model
    loss, accTrain = model.evaluate(X_train, y_train, verbose=0)
    print('Train Accuracy: %.3f' % accTrain)
    loss, accTest = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % accTest)

    # make a prediction with a new example
    fileName = "./KratosProject/MP1/sat42709.txt"
    inFileTester = open(fileName, 'r')
    tf = read_csv(inFileTester, header=None)
    testRows = tf.values[:, 2:-1]

    row = testRows[1]
    row = np.asarray(row).astype('float32')

    yhat = model.predict([[row]])
    #print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
    print("Train:", accTrain, "Test", accTest)
    accuracyTotal = (accTrain + accTest)/2
    return accuracyTotal


activationList = ['sigmoid', 'softmax']
nodeCountList = [6, 12, 24, 48, 72]
testSizeList = [.33, .25, .1]

# Best combination
MLP('sigmoid', 'softmax', 6, 24, 0.1, 50)

# Find best combo
bestAcc = -1

'''
for firstActivation in activationList:
    for secondActivation in activationList:
        for firstNode in nodeCountList:
            for secondNode in nodeCountList:
                for testSize in testSizeList:
                    currentAccuracy = MLP(
                        firstActivation, secondActivation, firstNode, secondNode, testSize, 10)
                    if (currentAccuracy > bestAcc):
                        bestAcc = currentAccuracy
                        bestAccList = [
                            firstActivation, secondActivation, firstNode, secondNode, testSize]

print(bestAccList)
print(bestAcc)
'''
