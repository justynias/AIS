from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree


iris = load_iris()
train, test, train_labels, test_labels = train_test_split(iris.data[:, :2], iris.target, test_size=0.50, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train, train_labels)

print("Classification Efficiency:", clf.score(test, test_labels))