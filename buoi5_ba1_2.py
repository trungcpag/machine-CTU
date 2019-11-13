import random
import numpy as np 
 
def my_perceptron(X, y, eta, lanlap):
    n = len(X[0,])
    m = len(X[:, 0])
    print("m =",m, " va n = ", n)
    w0 = -0.2
    w = (0.5, 0.5)
    print(" w0 = ", w0)
    print(" w = ", w)
    for t in range(0, lanlap):
        print("Lan lan: ", t)
        for i in range(0, m):
            gx = w0 + sum(X[i, ]*w)
            print("gx = ", gx)
            if(gx > 0):
                output = 1
            else:
                output = 0
            w0 = w0 + eta*(y[i] - output)
            w = w + eta*(y[i] - output)*X[i,]
            print(" w0 = ", w0)
            print(" w = ", w)
    return (w0 ,w)

X = np.array([[0,0,1,1],[0,1,0,1]])
Y = np.array([0,0,0,1])
X = X.T
my_perceptron(X, Y, 0.15, 2)