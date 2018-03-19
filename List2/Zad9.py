import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


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

best_score=0
best_k=0
for k in range (1,20):
    neigh = KNeighborsClassifier(n_neighbors=k)
    #Training the Classifier
    neigh.fit(train, train_targets)
    #Testing
    Z=neigh.predict(test)
    temp_score=neigh.score(test, test_targets)
    if(temp_score>best_score):
        best_score=temp_score
        best_k=k
        
print("The best score:", best_score, "for k parameter:", best_k)
    

