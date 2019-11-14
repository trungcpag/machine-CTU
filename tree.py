import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
import graphviz
from graphviz import Source
from sklearn.tree import DecisionTreeClassifier

pima = pd.read_csv("H:/test.csv")
pima.head(8)

feature_cols = ['Sector_score', 'Score_A', 'Score_B', 'TOTAL','Money_Value']
X = pima[feature_cols] # Features
y = pima.Risk # Target variable
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

tree.export_graphviz(clf, out_file="H:/tree.dot",
                         feature_names=feature_cols,
                         class_names=['0','1'],
                         filled=True, rounded=True)