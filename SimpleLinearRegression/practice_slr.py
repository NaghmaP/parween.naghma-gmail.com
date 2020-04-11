# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:55:16 2020

@author: Naghma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#split the data into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#Fitting linear regression to training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predecting the test result
y_pred = regressor.predict(X_test)

#visualize training data
plt.scatter(X_train, y_train, color='Red')
plt.plot(X_train, regressor.predict(X_train), color='Blue')
plt.title("Salary v'/s Experience (Training Data)")
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#visualize Test data
plt.scatter(X_test, y_test, color='Red')
plt.plot(X_test, y_pred, color='Blue')
plt.title("Salary v'/s Experience (Test Data)")
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

