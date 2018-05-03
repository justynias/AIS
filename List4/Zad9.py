from sklearn import datasets
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
import math
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

mnist_dataset = datasets.load_digits()
X = mnist_dataset.data
Y = mnist_dataset.target
target_names = mnist_dataset.target_names
train, test, train_targets, test_targets = model_selection.train_test_split(X, Y, train_size=0.5,test_size=0.5)

knn = KNeighborsClassifier(round(math.sqrt(train.shape[0]+test.shape[0])))

sffs = SFS(knn, k_features=round(train.shape[1]*0.05), forward=True,floating=True,scoring="accuracy", cv=0)
sffs = sffs.fit(train, train_targets)


best_k_features=round(train.shape[1]*0.05)
best_score=sffs.k_score_
features=2
for i in range (1,5):
    features=features*i
    sffs = SFS(knn, k_features=features, forward=True,floating=True,scoring="accuracy", cv=0)
    sffs = sffs.fit(train, train_targets)
    #print("For number of featres: {0}, best features: {1}, prediction score: {2}".format(features, sffs.k_feature_idx_, sffs.k_score_))
    if best_score<sffs.k_score_:
        best_score=sffs.k_score_
        best_k_features=features
    
print("The best score: {0} for number of features: {1}".format(best_score,best_k_features ))