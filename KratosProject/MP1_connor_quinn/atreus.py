from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# load the dataset
fileName = "KratosProject/MP1_connor_quinn/sat41589"
inFile = open(fileName, 'r')
df = read_csv(inFile, header=None)
print (df)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
X = X.astype('float32')

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# determine the number of input features
n_features = X_train.shape[1]
print("n=", n_features)

# define model
model = Sequential()
model.add(Dense(5, activation='sigmoid',
          kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(3, activation='softmax'))

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
#model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(X_train, y_train, epochs=5000, batch_size=50, verbose=2)

# evaluate the model
loss, acc = model.evaluate(X_train, y_train, verbose=0)
print('Train Accuracy: %.3f' % acc)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

# make a prediction with a new example
row = [5.1, 3.5, 1.4, 0.2]
yhat = model.predict([row])
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
