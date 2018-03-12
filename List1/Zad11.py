import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import SubplotZero

#definicje funkcji klasyfikujących
def g1(x):
    return -x[0]+x[1]

def g2(x):
    return x[0]-x[1]

#klasyfikator
def classifier(x):
    if g1(x)>g2(x):
        return 1;
    else:
        return 2;
#powierzchnia decyzyjna
#-x[0]+x[1]=x[0]-x[1]
# x[0]=x[1]

fig = plt.figure("Powierzchnia deycyzjna")
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)
for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
        ax.axis[direction].set_visible(False)

ax.plot([-10,10],[-10, 10])
plt.show()


