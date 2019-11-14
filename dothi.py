import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('audit_risk.csv')
#sns.countplot(df['Risk'], label = "Count")
#plt.show()
columns = df.iloc[:, [0,1,2,3,5,6,8,9,12,18,21]]
l = len(df.Sector_score.unique())
for j in range (0,l):
    print(df.Sector_score.unique()[j] + ":" + df.Sector_score.unique()[j].count(level))
