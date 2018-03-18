import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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


gnb = GaussianNB()

#Training the Classifier
clf = gnb.fit(train, train_targets)

#Testing
Z = clf.predict(test)



class1=test[(Z==1).nonzero()]
class2=test[(Z==2).nonzero()]


plt.scatter(class1[:,0], class1[:,1],c='b')
plt.scatter(class2[:,0], class2[:,1],c='r')
plt.show()

# decisions field
# how do we know c,h ??
C = 1.0
h = .02
x_min, x_max = test[:, 0].min() - 1, test[:, 0].max() + 1
y_min, y_max = test[:, 1].min() - 1, test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
plt.show()