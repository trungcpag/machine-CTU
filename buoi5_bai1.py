import numpy as np 
import matplotlib.pyplot as plt

X = np.array([[0,0,1,1],[0,1,0,1]])
Y = np.array([0,0,0,1])
X = X.T
X
colormap = np.array(["red", "green"])
plt.axis([0, 1.2, 0, 1.2])
plt.scatter(X[:, 0], X[:,1], c = colormap[Y[:]])
plt.xlabel("Gia tri X1")
plt.xlabel("Gia tri X2")
plt.show()