# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 09:57:52 2019

@author: suraj
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.set_printoptions(threshold=np.nan)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

def load_housing_data(housing_path,fileName):
    csv_path = os.path.join(housing_path, fileName)
    return pd.read_csv(csv_path)

train = load_housing_data("D:\GitRepository\Machine-Learning\HousePrediction\house-prices-advanced-regression-techniques", "train.csv")
test = load_housing_data("D:\GitRepository\Machine-Learning\HousePrediction\house-prices-advanced-regression-techniques", "test.csv")

train.shape
test.shape
train.head()

# Getting rid of the IDs but keeping the test IDs in a variable
#Save the 'Id' column
trainID = train["Id"]
testID = test["Id"]

#drop the ID Column
train.drop("Id",axis=1,inplace= True)
test.drop("Id",axis=1,inplace= True)

na_cutoff = 0.2

train.SalePrice.describe()

plt.subplots(figsize=(10,7))
sns.distplot(train["SalePrice"])

corrmat = train.corr() #analyze the correlation between data and target variable

plt.subplots(figsize=(18,12))
sns.heatmap(train.corr())

#Observe strong co related features and taking them to explore relation with SalePrice
k = 5 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cols

cm = np.corrcoef(train[cols].values.T)
plt.subplots(figsize=(10,7))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']].corr()

sns.pairplot(data=train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']])

train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']].describe()

train.select_dtypes(include='object').describe()

#box plot overallqual/saleprice
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=train['OverallQual'], y=train["SalePrice"],)

f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=train['YearBuilt'], y=train["SalePrice"],)
plt.xticks(rotation=90)

train.info()

train.drop("Alley", axis = 1, inplace = True)
train.drop("FireplaceQu", axis = 1, inplace = True)
train.drop("PoolQC", axis = 1, inplace = True)
train.drop("Fence", axis = 1, inplace = True)
train.drop("MiscFeature", axis = 1, inplace = True)

test.drop("Alley", axis = 1, inplace = True)
test.drop("FireplaceQu", axis = 1, inplace = True)
test.drop("PoolQC", axis = 1, inplace = True)
test.drop("Fence", axis = 1, inplace = True)
test.drop("MiscFeature", axis = 1, inplace = True)

# Seperating the output label
train_Labels = train["SalePrice"]
train.drop("SalePrice",axis = 1, inplace = True)

# 1. (DataCleanUp) Replace nan value
class DataFrameImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
# Imputing the training data
train_temp = DataFrameImputer().fit_transform(train)

# Imputing the test data
test_temp = DataFrameImputer().fit_transform(test)

#2. (DataCleanUp) Handling the Text and Categorical attributes. Convert Categorical data into numbers (i.e. LabelEncoding and OneHotEncoding)
train_tr = pd.get_dummies(train_temp)

# Handling the categorical data of test
test_tr = pd.get_dummies(test_temp)

non_matching_col =  list(set(list(train_tr)) ^ set(list(test_tr)))

for col in non_matching_col:
    if(col in train_tr.columns):
        train_tr.drop(col, axis = 1, inplace = True)
    if(col in test_tr.columns):
        train_tr.drop(col, axis = 1, inplace = True)

#3. (DataCleanUp) Feature Scaling. Standard Scaling the dataset i.e. converting the dataset value in between -1 to 1
standardScalerX = StandardScaler()
train_sclr = standardScalerX.fit_transform(train_tr)

# Scaling the test data set
test_sclr = standardScalerX.fit_transform(test_tr)

# Training the Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_sclr, train_Labels)

# Predict the output
predict = lin_reg.predict(test_sclr);
