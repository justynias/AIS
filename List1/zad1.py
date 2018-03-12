file=open("iris.txt", 'r')

iris_tab=[]
i=0

for line in file:
    row=(line.strip()).split(",")
    iris_tab.append(row)

print("Liczba probek: "+ str(len(iris_tab)))
for probe in iris_tab:
    i=i+1
    print("ilosc atrybutow w probce nr "+ str(i)+": "+ str(len(row)))

