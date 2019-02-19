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
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
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

#na_cutoff = 


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

train_num = train.copy()

train_num.drop("MSZoning",axis = 1, inplace = True)
train_num.drop("Street",axis = 1, inplace = True)
train_num.drop("LotShape",axis = 1, inplace = True)
train_num.drop("LandContour",axis = 1, inplace = True)
train_num.drop("Utilities",axis = 1, inplace = True)
train_num.drop("LotConfig",axis = 1, inplace = True)
train_num.drop("LandSlope",axis = 1, inplace = True)
train_num.drop("Neighborhood",axis = 1, inplace = True)
train_num.drop("Condition1",axis = 1, inplace = True)
train_num.drop("Condition2",axis = 1, inplace = True)
train_num.drop("BldgType",axis = 1, inplace = True)
train_num.drop("HouseStyle",axis = 1, inplace = True)
train_num.drop("RoofStyle",axis = 1, inplace = True)
train_num.drop("RoofMatl",axis = 1, inplace = True)
train_num.drop("Exterior1st",axis = 1, inplace = True)
train_num.drop("Exterior2nd",axis = 1, inplace = True)
train_num.drop("MasVnrType",axis = 1, inplace = True)
train_num.drop("ExterQual",axis = 1, inplace = True)
train_num.drop("ExterCond",axis = 1, inplace = True)
train_num.drop("Foundation",axis = 1, inplace = True)
train_num.drop("BsmtQual",axis = 1, inplace = True)
train_num.drop("BsmtCond",axis = 1, inplace = True)
train_num.drop("BsmtExposure",axis = 1, inplace = True)
train_num.drop("BsmtFinType1",axis = 1, inplace = True)
train_num.drop("BsmtFinType2",axis = 1, inplace = True)
train_num.drop("Heating",axis = 1, inplace = True)
train_num.drop("HeatingQC",axis = 1, inplace = True)
train_num.drop("CentralAir",axis = 1, inplace = True)
train_num.drop("Electrical",axis = 1, inplace = True)
train_num.drop("KitchenQual",axis = 1, inplace = True)
train_num.drop("Functional",axis = 1, inplace = True)
train_num.drop("GarageType",axis = 1, inplace = True)
train_num.drop("GarageFinish",axis = 1, inplace = True)
train_num.drop("GarageQual",axis = 1, inplace = True)
train_num.drop("GarageCond",axis = 1, inplace = True)
train_num.drop("PavedDrive",axis = 1, inplace = True)
train_num.drop("SaleType",axis = 1, inplace = True)
train_num.drop("SaleCondition",axis = 1, inplace = True)

## 1. Data Cleaning Removing the nan data from the numerical columns
#imputer = Imputer(strategy="median")
#imputer.fit(train_num) #training the imputer 
##using the trained imputer to replace the missing values
#X = imputer.transform(train_num)
#
##train_bk = train
##train = train_bk
#
##putting back the imputed data to the pandas dataframe i.e data replaced with the num
#train_tr = pd.DataFrame(X, columns=train_num.columns)

#imputer.statistics_

# 2. Handling text and categorical Attributes
train_cat = train.copy()

train_cat.drop("MSSubClass",axis = 1, inplace = True)
train_cat.drop("LotFrontage",axis = 1, inplace = True)
train_cat.drop("LotArea",axis = 1, inplace = True)
train_cat.drop("OverallQual",axis = 1, inplace = True)
train_cat.drop("OverallCond",axis = 1, inplace = True)
train_cat.drop("YearBuilt",axis = 1, inplace = True)
train_cat.drop("YearRemodAdd",axis = 1, inplace = True)
train_cat.drop("MasVnrArea",axis = 1, inplace = True)
train_cat.drop("BsmtFinSF1",axis = 1, inplace = True)
train_cat.drop("BsmtFinSF2",axis = 1, inplace = True)
train_cat.drop("BsmtUnfSF",axis = 1, inplace = True)
train_cat.drop("TotalBsmtSF",axis = 1, inplace = True)
train_cat.drop("1stFlrSF",axis = 1, inplace = True)
train_cat.drop("2ndFlrSF",axis = 1, inplace = True)
train_cat.drop("BsmtHalfBath",axis = 1, inplace = True)
train_cat.drop("LowQualFinSF",axis = 1, inplace = True)
train_cat.drop("GrLivArea",axis = 1, inplace = True)
train_cat.drop("BsmtFullBath",axis = 1, inplace = True)
train_cat.drop("FullBath",axis = 1, inplace = True)
train_cat.drop("HalfBath",axis = 1, inplace = True)
train_cat.drop("BedroomAbvGr",axis = 1, inplace = True)
train_cat.drop("KitchenAbvGr",axis = 1, inplace = True)
train_cat.drop("TotRmsAbvGrd",axis = 1, inplace = True)
train_cat.drop("Fireplaces",axis = 1, inplace = True)
train_cat.drop("GarageYrBlt",axis = 1, inplace = True)
train_cat.drop("GarageCars",axis = 1, inplace = True)
train_cat.drop("GarageArea",axis = 1, inplace = True)
train_cat.drop("WoodDeckSF",axis = 1, inplace = True)
train_cat.drop("OpenPorchSF",axis = 1, inplace = True)
train_cat.drop("EnclosedPorch",axis = 1, inplace = True)
train_cat.drop("3SsnPorch",axis = 1, inplace = True)
train_cat.drop("ScreenPorch",axis = 1, inplace = True)
train_cat.drop("PoolArea",axis = 1, inplace = True)
train_cat.drop("MiscVal",axis = 1, inplace = True)
train_cat.drop("MoSold",axis = 1, inplace = True)
train_cat.drop("YrSold",axis = 1, inplace = True)
train_cat.drop("SalePrice",axis = 1, inplace = True)

import category_encoders as ce
# Get a new clean dataframe
train_cat = train.select_dtypes(include=['object']).copy()

# Specify the columns to encode then fit and transform
encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=["MSZoning",
                                                                 "Street",
                                                                 "LotShape",
                                                                 "LandContour",
                                                                 "Utilities",
                                                                 "LotConfig",
                                                                 "LandSlope",
                                                                 "Neighborhood",
                                                                 "Condition1",
                                                                 "Condition2",
                                                                 "BldgType",
                                                                 "HouseStyle",
                                                                 "RoofStyle",
                                                                 "RoofMatl",
                                                                 "Exterior1st",
                                                                 "Exterior2nd",
                                                                 "MasVnrType",
                                                                 "ExterQual",
                                                                 "ExterCond",
                                                                 "Foundation",
                                                                 "BsmtQual",
                                                                 "BsmtCond",
                                                                 "BsmtExposure",
                                                                 "BsmtFinType1",
                                                                 "BsmtFinType2",
                                                                 "Heating",
                                                                 "HeatingQC",
                                                                 "CentralAir",
                                                                 "Electrical",
                                                                 "KitchenQual",
                                                                 "Functional",
                                                                 "GarageType",
                                                                 "GarageFinish",
                                                                 "GarageQual",
                                                                 "GarageCond",
                                                                 "PavedDrive",
                                                                 "SaleType",
                                                                 "SaleCondition"
                                                                 ])
encoder.fit(train_cat, verbose=1)

train_tr = encoder.transform(train_cat)

from sklearn.base import BaseEstimator, TransformerMixin

#This will transform the data by selecting the desired attributes, 
#dropping the rest and converting the resulting DataFrame to NumPy array.
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_attribs = list(train_num)
cat_attribs = list(train_cat)

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler',StandardScaler()),
        ])
    
#cat_pipeline = Pipeline([
#        ('selector', DataFrameSelector(cat_attribs)),
#        ('label_binarizer',MyLabelBinarizer()),
#        ])
    
    
#A Full pipeline handling both the numerical and categorical attributes

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list = [
        ("num_pipeline", num_pipeline)
        ])
    
#     ("cat_pipeline", cat_pipeline),
        
train_prepared = full_pipeline.fit_transform(train)

index = [i for i in range(0, len(train_prepared))]

df = pd.DataFrame(data=train_prepared, index= index)

train_Full = train_tr.append(df, ignore_index = True)