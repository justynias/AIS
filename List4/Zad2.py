#2. Korzystajączfunkcjisklearn.datasets.fetch_mldatapobierzzbiórdanychMNIST.
#Zbiór ten zawiera zdigitalizowane próbki ręczne pisma cyfr od 0 do 9. 
#Podziel zbiór losowo na część uczącą i testową.
from sklearn import datasets
from sklearn import model_selection
 
mnist_dataset = datasets.load_digits()
X = mnist_dataset.data
Y = mnist_dataset.target
target_names = mnist_dataset.target_names
train, test, train_targets, test_targets = model_selection.train_test_split(X, Y, train_size=0.5, test_size=0.5)