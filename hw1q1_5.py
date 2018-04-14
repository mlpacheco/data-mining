'''
Maria L. Pacheco
pachecog@purdue.edu
'''

from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np

df = pd.read_csv(r'creditcard.csv')
X = df.drop(columns=['Time', 'Class']).values
y_gold = df.loc[:, 'Class'].values

unique, counts = np.unique(y_gold, return_counts=True)
print("Class stats:", counts)

model = LogisticRegression(C=10e-8)
model.fit(X, y_gold)
y_pred = model.predict(X)


CM = confusion_matrix(y_gold, y_pred)
acc_pos = CM[1,1]/(CM[1,0]+CM[1,1])
acc_neg = CM[0,0]/(CM[0,0]+CM[0,1])
print("Accuracy of positive class:",acc_pos)
print("Accuracy of negative class:",acc_neg)

