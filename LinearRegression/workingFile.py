# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 01:46:54 2024

@author: Dimitrios Toliopoulos
"""

#general
import io

# data
import numpy as np
import pandas as pd

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# data visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt



column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight',
                'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv(r'G:\Coding\GitHub\LearningML\TrainingLinearRegression\abalone\abalone.data', 
                 names = column_names, sep = ',')
print(df)
print(df.columns)


# Google mini Questions
# Part 2 //////////////////////////////////////////////////////////////////////
# 1) What is the maximum whole weight?

max_weight = df['Whole_weight'].max()

# 2) What is the mean diameter across all trips?

mean_diameter = df['Diameter'].mean()

# 3) How many sexes are in the dataset?

count_sex = df['Sex'].nunique()

# 4) What is the most frequent sex?

most_freq_sex = df['Sex'].value_counts().idxmax()

# 5) Are any features missing data?

missing_values = df.isnull().sum().sum()

# Correlation matrix between features (and plot)

correlation_matrix = df.corr(numeric_only = True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Visualize relationships using pairplot

sns.pairplot(df[["Length", "Height", "Whole_weight"]])
plt.show()

#//////////////////////////////////////////////////////////////////////////////

# Get rid of outliers using Z-scores
from scipy.stats import zscore

df_numeric = df.select_dtypes(include=[np.number])
z_scores = np.abs(zscore(df_numeric))
df = df[(z_scores < 3).all(axis=1)]


# Creating Linear Regression model
X = df['Height'].values
y = df['Length'].values
X = X.reshape(-1,1)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mean_squared_error(y, y_pred)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('Abalone Height')
plt.ylabel('Abalone Length')
plt.title('Linear Regression')
plt.legend()
plt.show()

