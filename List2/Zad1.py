import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split

banana = sio.loadmat("banana.mat")
train_data = banana["train_data"]
train_labels = banana["train_labels"]
train_labels = np.array(train_labels)
test_data = banana["test_data"]
test_labels = banana["test_labels"]
test_labels = np.array(test_labels)

data=np.concatenate((train_data,test_data),axis=0)
data_labels=np.concatenate((train_labels, test_labels), axis=0)

# divisionf of the whole set, training 30%, testing 70%
train, test, train_labels1, test_labels1 = train_test_split(data, data_labels.ravel(), test_size=0.70)