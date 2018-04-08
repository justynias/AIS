from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree

from sklearn.externals.six import StringIO  
import pydot

iris = load_iris()
train, test, train_labels, test_labels = train_test_split(iris.data, iris.target, test_size=0.50, random_state=42)

clf = tree.DecisionTreeClassifier(min_samples_leaf=15, max_leaf_nodes=12)
clf = clf.fit(train, train_labels)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("decision_tree_min_max.pdf")
