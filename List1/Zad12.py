import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import SubplotZero

#definicje funkcji klasyfikujących
def g1(x):  #druga ćwiartka
    return -x[0]+x[1]

def g2(x):  #czwarta ćwiartka
    return x[0]-x[1]

#klasyfikator
def classifier(x):
    if g1(x)>g2(x):
        return 1;
    else:
        return 2;

N = 10

x1= 4*np.random.randn(N,2)+(-2)
x2= 4*np.random.randn(N,2)+(+2)

fig = plt.figure(1)
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)
for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
        ax.axis[direction].set_visible(False)

ax.plot(x1[:,0],x1[:,1],'yo', x2[:,0],x2[:,1], 'go')
ax.plot([-10,10],[-10, 10])

data=np.vstack((x1, x2))
print(data)
# #wartości funkcji klasyfikujących
# y1=[g1(data[i,:]) for i in range(2*N)]
# y2=[g2(data[i,:]) for i in range(2*N)]
#
# print("Wartości funkcji klasyfikujacych")
# print(y1)
# print(y2)

#decyzje klasyfikatora
decisions=np.array([classifier(data[i,:]) for i in range(2*N)])
print("Numery klas dla powyższych danych według klasyfikatora:")
print(decisions)

#indeksy próbek zakalsyfikowanych do danych klas
class1 = (decisions==1).nonzero()
class2 = (decisions==2).nonzero()

fig = plt.figure(2)
ax2 = SubplotZero(fig, 111)
fig.add_subplot(ax2)
for direction in ["xzero", "yzero"]:
        ax2.axis[direction].set_axisline_style("-|>")
        ax2.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
        ax2.axis[direction].set_visible(False)

ax2.scatter(data[class1,0],data[class1,1],c='b')
ax2.scatter(data[class2,0],data[class2,1],c='r')

ax2.plot([-10,10],[-10, 10])
plt.show()
