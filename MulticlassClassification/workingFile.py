# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:48:10 2024

@author: Dimitrios Toliopoulos
"""

# Multiclass classification using NN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt

dataset = pd.read_excel(r'G:\Coding\GitHub\LearningML\MulticlassClassification\DryBeanDataset\Dry_Bean_Dataset.xlsx')
df = dataset[['Perimeter', 'MajorAxisLength', 'ConvexArea', 'EquivDiameter', 'Extent', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class']]

# Data visualization
df.describe()
#df.hist(figsize=(16, 12), bins=20)

# Preparing data
X = df[['ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']]
y = df[['Class']]
X.hist(figsize=(16, 12), bins=20)

# Convert the string labels into numerical values
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse_output=False)
y = label_encoder.fit_transform(y.values.ravel())
y = one_hot_encoder.fit_transform(y.reshape(-1, 1))

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# One-Versus-All (OvA) model
def model_creation():
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = model_creation()
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate accuracy(2 ways)
accuracy = np.mean(y_pred_classes == y_test_classes)
print(f'OvA Test Accuracy: {accuracy}')
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))

# Direct comparison of predictions and true labels
comparison_df = pd.DataFrame({
    'True Label': label_encoder.inverse_transform(y_test_classes),
    'Predicted Label': label_encoder.inverse_transform(y_pred_classes)
})

#//////////////////////////////////////////////////////////////////////////////
# One-Versus-One (OvO) model
# Preparing data
X = df[['ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']]
y = df[['Class']]

# Convert the string labels into numerical values
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse_output=False)
y = label_encoder.fit_transform(y.values.ravel())
y = one_hot_encoder.fit_transform(y.reshape(-1, 1))

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def create_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification for OvO
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Wrap the model with KerasClassifier for use in scikit-learn
ovo_classifier = OneVsOneClassifier(KerasClassifier(model=create_model, epochs=10, batch_size=10, verbose=1))


ovo_classifier.fit(X_train, y_train)
y_pred = ovo_classifier.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f'OvO Test Accuracy: {accuracy}')

'''
Using a One-Versus-One (OvO) model creates a separate binary classifier for 
each pair of classes. With 7 unique classes, the OvO model will train 21 binary 
classifiers, each distinguishing between two different classes. This approach 
significantly increases the computational complexity compared to One-Versus-All 
(OvA) since the model has to train and evaluate multiple classifiers. While 
OvO can sometimes lead to higher accuracy in certain problems (especially when 
class distinctions are more localized), the overall improvement in accuracy may 
not always justify the additional computational overhead. The increase in 
processing power required can be substantial, particularly when there are many 
classes.
'''