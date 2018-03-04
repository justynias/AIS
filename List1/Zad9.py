import numpy as np
from sklearn import metrics

N = 10
x1 = 1*np.random.randn(N)+(-2)
x2 = 10*np.random.rand(N)

data=np.vstack((x1,x2))
data=data.conj().transpose()

eukl_matrix=metrics.pairwise.pairwise_distances(data, metric='euclidean')
print("Macierz odległości euklidesowych: ")
print(eukl_matrix)

mahal_matrix=metrics.pairwise.pairwise_distances(data, metric='mahalanobis')
print("Macierz odległości mahalanobisa: ")
print(mahal_matrix)

mink_matrix=metrics.pairwise.pairwise_distances(data, metric='minkowski')
print("Macierz odległości Minkowskiego: ")
print(mink_matrix)
