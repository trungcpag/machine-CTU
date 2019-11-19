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
# print(data_test)
lab_test = trial.iloc[0:772, 17]
trial['Risk'].head(200)
sum = 0.0
workbook = xlwt.Workbook() 