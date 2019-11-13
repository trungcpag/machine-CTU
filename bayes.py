import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
df=pd.read_csv('audit_risk.csv')
print(df.describe())
data = df.iloc[:, [0,1,2,3,5,6,8,9,12,18,21]]
lab = df['Risk']

trial=pd.read_csv('trial.csv')
data_test = trial.iloc[0:772, [0,1,2,3,4,5,6,7,9,14,16]]
lab_test = trial.iloc[0:772, 17]
model=GaussianNB()
model.fit(data,lab)
dubao=model.predict(data_test)
print(accuracy_score(lab_test,dubao)*100)