# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 13:55:44 2018

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]#Take all rows and start from column one take all column (i.e excluding 0th or first column)

#Encoding categorical data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting the multiple linear regression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set
y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
def backwardElimination(x, sl, y_val):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = y_val,exog = x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for jval in range(0, numVars - 1):
                if regressor_OLS.pvalues[jval].astype(float) == maxVar:
                    x = np.delete(x, jval, axis=1)
                    break
    regressor_OLS.summary()
    return x



SL = 0.05
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
y_val = y
X_Modeled = backwardElimination(X_opt,SL, y_val)


