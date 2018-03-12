from sklearn import preprocessing
import numpy as np

file=open("iris.txt", 'r')

iris_tab=[]
i=0

sep_len=[]
sep_wid=[]
pet_len=[]
pet_wid=[]

for line in file:
    row = (line.strip()).split(",")
    sep_len.append(float(row[0]))
    sep_wid.append(float(row[1]))
    pet_len.append(float(row[2]))
    pet_wid.append(float(row[3]))

data=np.vstack((sep_len,sep_wid,pet_len,pet_wid))

#normalizacja
data_norm = preprocessing.normalize(data, norm="l1")

def show_normalized(atribute, index):
    print("---- "+ atribute + " ----")
    print("Wartosc minimalna: " + str(min(data[index,:])))
    print("Wartosc maksymalna: " + str(max(data[index,:])))
    print("Wartosc srednia: " + str(np.mean(data[index,:])))
    print("Wartosc odchylenia standardowego: " + str(np.std(data[index,:])))
    print("PO NORAMLIZACJI")
    print("Wartosc minimalna: " + str(min(data_norm[index,:])))
    print("Wartosc maksymalna: " + atribute + ": " + str(max(data_norm[index,:])))
    print("Wartosc srednia: " + str(np.mean(data_norm[index,:])))
    print("Wartosc odchylenia standardowego: " + str(np.std(data_norm[index,:]))+ "\n")


show_normalized("Sepal length", 0)
show_normalized("Sepal width", 1)
show_normalized("Petal length", 2)
show_normalized("Petal width", 3)


