# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:40:37 2024

@author: Dimitrios Toliopoulos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, precision_recall_curve

df = pd.read_excel(r'G:\Coding\GitHub\LearningML\BinaryClassification\Raisin_Dataset\Raisin_Dataset.xlsx')

'''
Many ways to create the features dataset 'X'
X = df.iloc[:, :-1]
X = df.drop('Target', axis=1)
X = df.loc[:, df.columns != 'Target']
'''

X = df.drop(['Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}\n\nConfusion Matrix:\n{conf_matrix}")
print(class_report)

#//////////////////////////////////////////////////////////////////////////////
# Binary Classification Part 2
# Using a second dataset with positive and negative labels.

dataset = pd.read_csv(r'G:\Coding\GitHub\LearningML\BinaryClassification\breast_cancer_wisconsin_diagnostic\wdbc.data', header=None)
dataset[1] = dataset[1].replace({'M':1, 'B':0})
dataset[1].value_counts()

X = dataset.drop([1], axis=1)
y = dataset[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

# Adjusting the decision threshold analyzing the trade-offs between recall and precision.

y_probs = model.predict_proba(X_test)[:, 1]
#y_pred = model.predict(X_test)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for threshold in thresholds:
    y_pred = (y_probs >= threshold).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f'Threshold: {threshold:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')


precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall vs. Threshold")
plt.legend(loc="best")
plt.show()

'''
In breast cancer detection, a higher recall reduces the risk of missing actual
cancer cases. Although this may lead to more false positives (lower precision),
the consequence of a false negative (missing a cancer case) is usually 
considered more severe in this context. In this example, a threshold value of 
0.5 provides us with an almost perfect value of recall and a decent level of 
precision.
'''


'''
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Accuracy: {accuracy}\n\nConfusion Matrix:\n{conf_matrix}\n\n"
      f"Recall: {recall}\n\nPrecision: {precision}")
'''
