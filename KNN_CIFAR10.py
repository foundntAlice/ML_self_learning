import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
'''
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for y, cls in enumerate(classes):
  idxs = np.flatnonzero(y_train)
  idxs = np.random.choice(idxs, 7, replace=False)
  for i, idx in enumerate(idxs):
    plt_idx = i * 10 + y + 1
    plt.subplot(7, 10, plt_idx)
    plt.imshow(X_train[idx].astype('uint8'))
    plt.axis('off')
    if i == 0:
      plt.title(cls)

plt.show()
'''
num_training = 20000
num_test = 1000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

#reshape
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

class KNN():
  def __init__(self):
    pass
  def train(self, X, y):
    self.X_tr = X
    self.y_tr = y
  def predict(self, X, k):
    num_test = X.shape[0]
    y_pred = np.zeros(num_test, dtype=self.y_tr.dtype)
    for i in range(num_test):
      distances = np.sum(np.abs(self.X_tr - X[i, :]), axis = 1)
      min_idx = np.argpartition(distances, k)[0:k]
      votes = self.y_tr[min_idx]
      votes = votes.flatten()
      labels_count = np.bincount(votes)
      y_pred[i] = np.argmax(labels_count)

    return y_pred
  
clf = KNN()
clf.train(X_train, y_train)
from sklearn.metrics import accuracy_score

#evaluate k (1~9)
for i in range(1, 10):
  res = clf.predict(X_test, i)
  print("i = ", i)
  print(accuracy_score(res, y_test))