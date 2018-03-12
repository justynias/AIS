import matplotlib.pyplot as plt

file=open("iris.txt", 'r')

iris_tab=[]
i=0

for line in file:
    row=(line.strip()).split(",")
    iris_tab.append(row)

sepal_len=[]
sepal_wid=[]

for probe in iris_tab:
    sepal_len.append(float(probe[0]))
    sepal_wid.append(float(probe[1]))

plt.plot(sepal_len, sepal_wid,'yo')
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()
