# From https://datascience.stackexchange.com/questions/26640/how-to-check-for-overfitting-with-svm-and-iris-data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

iris = load_iris()
X = iris.data[:, :4]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svm_model = svm.SVC(kernel='rbf', C=100, gamma='auto', probability=True)

svm_model.fit(X_train, y_train)
predictions = svm_model.predict(X_train)
print(accuracy_score(predictions, y_train))
predictions = svm_model.predict(X_test)
print(accuracy_score(predictions, y_test))

#scores = cross_val_score(svm_model, iris.data, iris.target, cv=15)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

###LOOP THROUGH THESE
#kernal = rbf, linear, and poly (2,3,4 degrees)
#get averages of apperient and true error rate

