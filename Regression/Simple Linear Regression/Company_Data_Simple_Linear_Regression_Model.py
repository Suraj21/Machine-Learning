# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 09:21:55 2019

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

companies = pd.read_csv("D:\GitRepository\Machine-Learning\Regression\Simple Linear Regression\Company_Data.csv")
X = companies.iloc[:,:-1].values
y = companies.iloc[:, 4].values

companies.head()
companies.corr()
sb.heatmap(companies.corr())


#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:] # Avoiding the dummy variable trap i.e. not including the first row

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting multiple linear regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#check for coefficient
print(regressor.coef_)

#check for intercept
print(regressor.intercept_)

#Check for R squared value

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
