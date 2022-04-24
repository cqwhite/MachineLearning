from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import numpy as np

# load the dataset
fileName = "./KratosProject/MP1_connor_quinn/sat41589.txt"
inFile = open(fileName, 'r')
df = read_csv(inFile, header=None)
# split into input and output columns
X, y = df.values[:, 2:-1], df.values[:, -1]
X = np.asarray(X).astype('float32')
y = np.asarray(y).astype('float32')
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# determine the number of input features
n_features = X_train.shape[1]
print("n=", n_features)

# define model
model = Sequential()
model.add(Dense(6, activation="sigmoid", input_shape=(n_features,)))
model.add(Dense(12, activation='softmax'))

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(X_train, y_train, epochs=500, verbose=2)

# evaluate the model
loss, acc = model.evaluate(X_train, y_train, verbose=0)
print('Train Accuracy: %.3f' % acc)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

# make a prediction with a new example
fileName = "./KratosProject/MP1_connor_quinn/sat42709.txt"
inFileTester = open(fileName, 'r')
tf = read_csv(inFileTester, header=None)
testRows = tf.values[:, 2:-1]

row = testRows[1]
row = np.asarray(row).astype('float32')

print(row)
yhat = model.predict([[row]])
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
