from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
train, test, train_labels, test_labels = train_test_split(iris.data, iris.target, test_size=0.50, random_state=42)
