#1. Wczytać zbiór olivetti faces. Dla liczby komponentów od 1 do 6 przeprowadzić 
#algorytmem PCA redukcję przestrzeni wymiaru cech. Zobrazować wyniki. 
#Dla jakiej liczby komponentów uzyskano najlepszy wynik?

from sklearn.decomposition import PCA
from sklearn import datasets 
 
olivetti = datasets.fetch_olivetti_faces()
X = olivetti.data
Y = olivetti.target
max = 0
max_n_components = 0

for i in range(1, 6):
     pca = PCA(n_components=i)
     X_r = pca.fit(X).transform(X)
     print(i, ":", pca.explained_variance_ratio_.sum())
     
     if max < pca.explained_variance_ratio_.sum():
         max = pca.explained_variance_ratio_.sum()
         max_n_components = i
         
print("The best result for: ", max_n_components)