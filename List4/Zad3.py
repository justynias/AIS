#Korzystając z algorytmu FLD dokonaj redukcji wymiaru cech dla różnej liczby cech zbioru uczącego MNIST.
#Następnie sprawdź sprawność klasyfikatora kNN dla zbioru testowego ograniczonego do wybranego 
#podzbioru cech. Parametr kk przyjmij jako pierwiastek z liczby obiektów w zbiorze. 
#Dla jakiej liczby cech osiągnięto najlepsze rezultaty?

from sklearn import datasets
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
import math 

mnist_dataset = datasets.load_digits()
X = mnist_dataset.data
Y = mnist_dataset.target
target_names = mnist_dataset.target_names
train, test, train_targets, test_targets = model_selection.train_test_split(X, Y, train_size=0.5,test_size=0.5)
                                                                            
max = 0
max_n_components = 0
for i in range(1, 10):
    lda = LDA(n_components=i)
    X_r = lda.fit(train, train_targets).transform(train)
    Y_r = lda.fit(test, test_targets).transform(test)
    clf = KNeighborsClassifier(round(math.sqrt(X.shape[0])),weights="uniform", metric="euclidean")
    clf.fit(X_r, train_targets)
    print(i, ":", clf.score(Y_r, test_targets))
    if max < clf.score(Y_r, test_targets):
        max = clf.score(Y_r, test_targets)
        max_n_components  = i
print("Best result for: ", max_n_components) 