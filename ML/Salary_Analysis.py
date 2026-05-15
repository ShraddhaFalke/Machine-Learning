# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 20:06:03 2026

@author: falke
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

Data = pd.read_csv(r'C:\Users\falke\VSCODE\ML\Salary_Data.csv')

x = Data.iloc[:, :-1]
y = Data.iloc[:, :-1]
Data.isnull().sum()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit (x_train, y_train)

y_pred = regressor.predict(x_test)

comparision = pd.DataFrame({'Actul':y_test, 'Prediction': y_pred})
print(comparision)

plt.scatter(x_test, y_test, color = 'Red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.tittle('Salary of Employee based on Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

Data                            

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_12yr_exper = m_slope * 12 + c_intercept
print(y_12yr_exper)

bias = regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

## Mean
Data.mean()
Data['Salary'].mean()
Data['YearsExperience'].mean()
## Median
Data.median()
Data['Salary'].median()
Data['YearsExperience'].median()
## Mode
Data.mode()

## Variance
Data.var()
Data['Salary'].var()
Data['YearsExperience'].var()

## Standard Deviation
Data.std()
Data['Salary'].std()
Data['YearsExperience'].std()

##Variation
from scipy.stats import variation
variation(Data.values)
variation(Data['Salary'])

## Corelation
Data.corr()
Data['Salary'].corr(Data['YearsExperience'])

## Skewness
Data.skew()
Data['Salary'].skew()

## Standard Error
Data.sem()
Data['Salary'].sem()
Data['YearsExperience'].sem()

## Z-Score
import scipy.stats as stats
Data.apply(stats.zscore)
stats.zscore(Data['Salary'])

## ANOVA
## Sum of Square Regression (SSR)
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

mean_total = np.mean(Data.values)
SST = np.sum((Data.values-mean_total)**2)
print(SST)

r_square = 1-(SSR/SST)
r_square

print(r_square)
print(bias)
print(variance)
