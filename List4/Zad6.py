#Pobierz zbiór danych Arcene ze strony https://archive.ics.uci.edu/ml/datasets/Arcene. 
#Podziel zbiór losowo na część uczącą i testową

from sklearn import model_selection
import scipy.io as sio

arcene = sio.loadmat("arcene.mat")
X=arcene["X"]
Y=arcene["y"]

train, test, train_targets, test_targets = model_selection.train_test_split(X, Y, train_size=0.5,test_size=0.5)