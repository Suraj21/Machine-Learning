# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:32:50 2019

@author: suraj
"""

#XGBoost - for gradient boosting and performance speed
# For Deep learning feature scaling is mandatory
# But for XGBoost feature scaling is not necessary, as it is a gradient boosting model with decision trees.
# In XGBoost all the interpretation of the model can be kept

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\GitRepository\Machine-Learning\DeepLearning\Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Applying K-fold cross validation
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= classifier, X = X_train, y = y_train, cv=10)
accuracies.mean()
accuracies.std()





