import numpy as np
file=open("iris.txt", 'r')

iris_tab=[]
i=0

for line in file:
    row = (line.strip()).split(",")
    iris_tab.append(row)

sepal_len_setosa=[]
petal_len_setosa=[]
sepal_wid_setosa=[]
petal_wid_setosa=[]
sepal_len_versicolor=[]
petal_len_versicolor=[]
sepal_wid_versicolor=[]
petal_wid_versicolor=[]

for probe in iris_tab:
    if probe[4] == "Iris-setosa":
        sepal_len_setosa.append(float(probe[0]))
        sepal_wid_setosa.append(float(probe[1]))
        petal_len_setosa.append(float(probe[2]))
        petal_wid_setosa.append(float(probe[3]))
    elif probe[4] == "Iris-versicolor":
        sepal_len_versicolor.append(float(probe[0]))
        sepal_wid_versicolor.append(float(probe[1]))
        petal_len_versicolor.append(float(probe[2]))
        petal_wid_versicolor.append(float(probe[3]))

print("***SETOSA***")
print("Sepal length: " + str(np.mean(sepal_len_setosa)) + " Sepal width: " + str(np.mean(sepal_wid_setosa)))
print("Petal length: " + str(np.mean(petal_len_setosa)) + " Petal width: " + str(np.mean(petal_wid_setosa)))

print("***VERSICOLOR****")
print("Sepal length: " + str(np.mean(sepal_len_versicolor)) + " Sepal width: " + str(np.mean(sepal_wid_versicolor)))
print("Petal length: " + str(np.mean(petal_len_versicolor)) + " Petal width: " + str(np.mean(petal_wid_versicolor)))
