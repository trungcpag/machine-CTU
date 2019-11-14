import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
import xlwt
df=pd.read_csv('audit_risk.csv')
data = df.iloc[:, [0,1,2,3,5,6,8,9,12,18,21]]
lab = df['Risk']
trial=pd.read_csv('trial.csv')
data_test = trial.iloc[0:772, [0,1,2,3,4,5,6,7,9,14,16]]
lab_test = trial.iloc[0:772, 17]
trial['Risk'].head(200)
sum = 0.0
workbook = xlwt.Workbook() 
sheet = workbook.add_sheet("Sheet Name") 
for i in range(1, 100):
    X_test70, X_test30, Y_test70, Y_test30 = train_test_split(data_test, lab_test, test_size = 4/10.0, random_state = i*10 )
    clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=20,min_samples_leaf=90)
    clf_entropy.fit(data,lab)
    data_pred = clf_entropy.predict(X_test70)
    sum += accuracy_score(Y_test70,data_pred)*100
    model=GaussianNB()
    model.fit(data,lab)
    dubao=model.predict(X_test70)
    net = Perceptron()
    net.fit(data,lab)
    dubaoPeceptron = net.predict(X_test70)
    # print("Tree:", accuracy_score(Y_test70,data_pred)*100)
    # print("Perceptron: ", accuracy_score(Y_test70, dubaoPeceptron) * 100)
    # print("BAYES: ", accuracy_score(Y_test70,dubao)*100)
    sheet.write(i, 1, accuracy_score(Y_test70,data_pred)*100)
    sheet.write(i, 2, accuracy_score(Y_test70,dubaoPeceptron)*100)
    sheet.write(i, 3, accuracy_score(Y_test70,dubao)*100)
workbook.save("sample.xls")