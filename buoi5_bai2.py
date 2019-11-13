import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
data = pd.read_csv("data_per.csv", delimiter=",")

X = data.iloc[:, 0:5]
Y = data.Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 3/10.0, random_state = 5)

from sklearn.linear_model import Perceptron

net = Perceptron()
net.fit(X_train, Y_train)
print(net)

# net.coef_ w   
# net.intercept_ w0 -24.
# net.n_iter_ 9
# unique (so phan tu khac nhau trong mang)
# value_ counts()
y_pred = net.predict(X_test)

from sklearn.metrics import accuracy_score
print("ti le : ", accuracy_score(Y_test, y_pred) * 100)
# 77.78%