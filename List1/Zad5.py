import matplotlib.pyplot as plt

file=open("iris.txt", 'r')

iris_tab=[]
i=0

for line in file:
    row=(line.strip()).split(",")
    iris_tab.append(row)



sepal_len_setosa=[]
petal_len_setosa=[]
sepal_len_versicolor=[]
petal_len_versicolor=[]
sepal_len_virginica=[]
petal_len_virginica=[]

for probe in iris_tab:
    if probe[4]=="Iris-setosa":
        sepal_len_setosa.append(float(probe[0]))
        petal_len_setosa.append(float(probe[2]))
    elif probe[4]=="Iris-versicolor":
        sepal_len_versicolor.append(float(probe[0]))
        petal_len_versicolor.append(float(probe[2]))
    else:
        sepal_len_virginica.append(float(probe[0]))
        petal_len_virginica.append(float(probe[2]))

plt.plot(sepal_len_setosa, petal_len_setosa, 'go')
plt.plot(sepal_len_versicolor, petal_len_versicolor, 'bo')
plt.plot(sepal_len_virginica, petal_len_virginica, 'ro')
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.show()
