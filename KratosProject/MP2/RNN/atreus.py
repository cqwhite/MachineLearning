import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
import numpy as np
from pandas import read_csv

# Parameter split_percent defines the ratio of training examples


def get_train_test(textFile, split_percent=0.8):
    df = read_csv(textFile, usecols=[1, 2, 3, 4, 5, 6, 7], engine='python')
    data = np.array(df.values.astype('float32'))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data).flatten()

    n = len(data)
    # Point for splitting data into train and test
    split = int(n*split_percent)
    train_data = data[range(split)]
    test_data = data[split:]
    return train_data, test_data, data

# Prepare the input X and target Y


def get_XY(dat, time_steps):
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    rows_x = len(Y)
    X = dat[range(time_steps*rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))
   # print('THIS IS THE Xs', X)
    # print("THIS IS THE Ys", Y)
    return X, Y


def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape,
              activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def print_error(trainY, testY, train_predict, test_predict):
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    # Print RMSE
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))

# Plot the result


def testRSME(testY, test_predict):
    return math.sqrt(mean_squared_error(testY, test_predict))


def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    #plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Anomaly Detection Scale')
    plt.title(
        'Actual and Predicted Values')
    plt.show()


def rnn(timeStep, timeStepUnit, hiddenUnit, denseUnit):
    trainFile = "./RNN/sat42709.txt"
    sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
    time_steps = timeStep
    train_data, test_data, data = get_train_test(trainFile)
    trainX, trainY = get_XY(train_data, time_steps)
    testX, testY = get_XY(test_data, time_steps)

    # Create model and train
    model = create_RNN(hidden_units=hiddenUnit, dense_units=denseUnit, input_shape=(time_steps, timeStepUnit),
                       activation=['tanh', 'tanh'])

    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

    # make predictions
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    print_error(trainY, testY, train_predict, test_predict)

    result = testRSME(testY, test_predict)

    plot_result(trainY, testY, train_predict, test_predict)

    return result


# Best Run
rnn(12, 1, 12, 1)
# best run 12, 1, 12, 1


def runTest():
    timeStepList = [12, 6, 3, 1]
    #timeStepUnitList = [1, 2, 3]
    hiddenUnitList = [3, 6, 1]
    #denseUnitList = [1, 2, 3]
    bestRMSE = 10000000
    bestRMSEList = []
    count = 0
    for timeStep in timeStepList:
        # for timeStepUnit in timeStepUnitList:
        for hiddenUnit in hiddenUnitList:
            # for denseUnit in denseUnitList:
            print("STARTING:",
                  timeStep, 1, hiddenUnit, 1)
            count += 1
            currentRMSE = rnn(timeStep, 1,
                              hiddenUnit, 1)
            if (currentRMSE < bestRMSE):
                bestRMSE = currentRMSE
                bestRMSEList = [timeStep, 1,
                                hiddenUnit, 1]
            print(count)
            f = open("rnnTestResults.txt", "a")
            f.write("\n")
            f.write("Test Round " + str(count) + "\n")
            f.write("Test List: TimeStep " + str(timeStep) + ", timeStepUnit " + str(1) + ", hiddenUnit " +
                    str(hiddenUnit) + ", denseUnit " + str(1) + "\n")
            f.write("Current RMSE " + str(currentRMSE) + "\n")
            f.write("Best List " + str(bestRMSEList) + "\n")
            f.write("Best RMSE " + str(bestRMSE) + "\n")
            f.write("-----------------------------------" + "\n")
            f.close()

    print("Best RSMELIST", bestRMSEList)
    print("RSME:", bestRMSE)
