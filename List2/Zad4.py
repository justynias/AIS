import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

error = np.mean(Z != test_targets)

print("the percentage of correct classifications: " +str(1-error))