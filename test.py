import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
df=pd.read_csv('audit_risk.csv')
print(df.describe())
data = df.iloc[:, [0,1,2,3,5,6,8,9,12,18,21]]
lab = df['Risk']
# print(data)
trial=pd.read_csv('trial.csv')
data_test = trial.iloc[0:772, [0,1,2,3,4,5,6,7,9,14,16]]
lab_test = trial.iloc[0:772, 17]
# trial['Risk'].head(200)
# print(data_test)
sum = 0.0
for i in range(1, 100):
    X_test70, X_test30, Y_test70, Y_test30 = train_test_split(data_test, lab_test, test_size = 3/10.0, random_state = i )
    clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=20,min_samples_leaf=90)
    clf_entropy.fit(data,lab)
    data_pred = clf_entropy.predict(X_test70)
    sum += accuracy_score(Y_test70,data_pred)*100
    # if(accuracy_score(Y_test70,data_pred)*100 >= 87):
    print(i, "accuracy_score Tree:", accuracy_score(Y_test70,data_pred)*100)
    model=GaussianNB()
    model.fit(data,lab)
    dubao=model.predict(X_test70)
    print("bAYES: ", accuracy_score(Y_test70,dubao)*100)
    
