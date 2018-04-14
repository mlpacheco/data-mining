from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(r'creditcard.csv')
X = df.drop(columns=['Time', 'Class']).values
y_gold = df.loc[:, 'Class'].values

unique, counts = np.unique(y_gold, return_counts=True)
print("Class stats:", counts)

Cs = np.asarray([10e-14, 10e-11, 10e-8, 10e-5, 10e-2, 10e2, 10e5, 10e8])
pos = []; neg = []
for C in Cs:
    print("using C:", C)
    model = LogisticRegression(C=C)
    model.fit(X, y_gold)
    y_pred = model.predict(X)


    CM = confusion_matrix(y_gold, y_pred)
    acc_pos = CM[1,1]/(CM[1,0]+CM[1,1])
    acc_neg = CM[0,0]/(CM[0,0]+CM[0,1])
    #print("Accuracy of positive class:",acc_pos)
    #print("Accuracy of negative class:",acc_neg)

    pos.append(acc_pos)
    neg.append(acc_neg)


plt.plot(Cs, pos, color='blue')
plt.plot(Cs, neg, color='red')
plt.xscale('log')
plt.xlabel("C")
plt.ylabel("accurcy")
plt.show()
