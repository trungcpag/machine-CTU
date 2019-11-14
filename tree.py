import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import tree
df=pd.read_csv('audit_risk.csv')
data = df.iloc[5:11,[0,1,2,3,21]]
lab = df.iloc[5:11,26]
print(data)
print(lab)
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(data,lab)
from sklearn.tree import export_graphviz
import graphviz 
from graphviz import Source
dotfile = open("D:/tree.dot", 'w')
# # dot_data = tree.export_graphviz(clf, out_file=dotfile, feature_names=data.columns,class_names=data,filled=True, rounded=True)
# # graph = graphviz.Source(dot_data) 
#  tree.export_graphviz(clf, out_file=dotfile,
#                          feature_names=data.columns,
#                          class_names=data,
#                          filled=True, rounded=True)
