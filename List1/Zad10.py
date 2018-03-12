import numpy as np
from sklearn import preprocessing
from sklearn import metrics

N = 10
x1 = 1*np.random.randn(N)+(-2)
x2 = 10*np.random.rand(N)

data=np.vstack((x1,x2))
data=data.transpose()

scale = preprocessing.MinMaxScaler((0,1))
#operacja skalowania
data_scaled = scale.fit_transform(data)

eukl_matrix=metrics.pairwise.pairwise_distances(data_scaled, metric='euclidean')
print("Macierz odległości euklidesowych: ")
print(eukl_matrix)

mahal_matrix=metrics.pairwise.pairwise_distances(data_scaled, metric='mahalanobis')
print("Macierz odległości mahalanobisa: ")
print(mahal_matrix)

mink_matrix=metrics.pairwise.pairwise_distances(data_scaled, metric='minkowski')
print("Macierz odległości Minkowskiego: ")
print(mink_matrix)
