from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot
import numpy as np

#Zzad 7,8
iris = load_iris()

train, test, train_labels, test_labels = train_test_split(iris.data, iris.target, test_size=0.50, random_state=42)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(train, train_labels)


dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("decision_tree_entropy.pdf")

y=clf.predict(test)

error = np.count_nonzero(y != test_labels)
print("Number of incorrect classifications:", error)
print("Classification Efficiency:", clf.score(test, test_labels))