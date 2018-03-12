import numpy as np
file=open("iris.txt", 'r')
iris_tab=[]
i=0

for line in file:
    row=(line.strip()).split(",")
    iris_tab.append(row)

sepal_len=[]
sepal_wid=[]
petal_len=[]
petal_wid=[]

for probe in iris_tab:
    sepal_len.append(float(probe[0]))
    sepal_wid.append(float(probe[1]))
    petal_len.append(float(probe[2]))
    petal_wid.append(float(probe[3]))

def analyzeData(atribute):
    print("Wartosc minimalna: "+ str(min(atribute)))
    print("Wartosc maksymalna: "+ str(max(atribute)))
    print("Srednia: "+ str(np.mean(atribute)))
    print("Odchylenie stanardowe: "+ str(np.std(atribute)))


print("****Wyniki obliczeń dla pierwszego atrybutu****")
analyzeData(sepal_len)
print("****Wyniki obliczeń dla drguego atrybutu****")
analyzeData(sepal_wid)
print("****Wyniki obliczeń dla trzeciego atrybutu****")
analyzeData(petal_len)
print("****Wyniki obliczeń dla czwartego atrybutu****")
analyzeData(sepal_wid)
