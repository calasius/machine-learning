__author__ = 'claudio'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import BernoulliRBM

train_file = open("MNIST_train.csv")
train_file.readline()
data_train = np.loadtxt(train_file, delimiter=',')

test_file = open("MNIST_test.csv")
test_file.readline()
data_test = np.loadtxt(test_file, delimiter=',')

X_train, X_test, y_train, y_test = train_test_split(data_train[:, 1:], data_train[:, 0], test_size=0.3, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

plt.imshow(X_train[0, :].reshape((28, 28)), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

classifier = BernoulliRBM()
classifier.fit(X_train_std, y_train)
y_pred = classifier.predict(X_test_std)
print('Miscalssified samples: %d' % (y_test != y_pred).sum())
print('Accuracy = %.2f' % accuracy_score(y_test, y_pred))
