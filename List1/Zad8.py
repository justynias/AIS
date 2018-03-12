import numpy as np
import matplotlib.pyplot as plt

N = 10
#próba losowa z rozkładu N(-2,1)
x1 = 1*np.random.randn(N)+(-2)

#próba losowa z rozkładu jednostajnego na przedziale [0,10]
x2 = 10*np.random.rand(N)

data=np.vstack((x1,x2))
data=data.transpose()

plt.scatter(data[:, 0], data[:, 1])
plt.xlabel("N(-2,1)Distribiution")
plt.ylabel("[0,10] Distribiution")
plt.show()

