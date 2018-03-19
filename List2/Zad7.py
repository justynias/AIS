import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid
import matplotlib.pyplot as plt

banana = sio.loadmat("banana.mat")
train_data = banana["train_data"]
train_labels = banana["train_labels"]
train_labels = np.array(train_labels)
test_data = banana["test_data"]
test_labels = banana["test_labels"]
test_labels = np.array(test_labels)

data=np.concatenate((train_data,test_data),axis=0)
data_labels=np.concatenate((train_labels, test_labels), axis=0)

# division of the whole set, training 30%, testing 70%
train, test, train_targets, test_targets = train_test_split(data, data_labels.ravel(), test_size=0.70, random_state=42)

#Training the Classifier
tmp = NearestCentroid()


tmp.fit(train, train_targets)

#Testing
Z = tmp.predict(test)


# Visualizing decision boundaries

c1 = (Z == 1).nonzero()
c2 = (Z == 2).nonzero()
plt.scatter(test[c1, 0], test[c1, 1], c="g", label="Class 1")
plt.scatter(test[c2, 0], test[c2, 1], c="r", label="Class 2")
plt.legend()
plt.scatter(tmp.centroids_[:, 0], tmp.centroids_[:, 1], c="b")
plt.show()
