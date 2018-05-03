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
sfs = SFS(knn, k_features=round(train.shape[1]*0.05),forward=True, floating=False, cv=0,scoring="accuracy")

sfs.fit(train, train_targets)
print("SFS score: ", sfs.k_score_)
 

