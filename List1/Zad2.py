import math

file = open("iris.txt", 'r')

iris_tab = []
i = 0

for line in file:
    row=(line.strip()).split(",")
    iris_tab.append(row)

print(iris_tab[75])
print(iris_tab[10])

pow_sum = 0

for i in range(3):

    pow_sum=pow_sum+(float(iris_tab[75][i])-float(iris_tab[10][i]))**2

print("Odleglosc wynosi: " + str(math.sqrt(pow_sum)))


