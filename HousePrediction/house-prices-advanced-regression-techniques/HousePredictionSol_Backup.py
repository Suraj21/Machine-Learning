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
#from sklearn.preprocessing import Imputer
#from sklearn.preprocessing import LabelBinarizer
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler

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

#train_num = train.copy()
#
#train_cat_col_list = ["MSZoning","Street","LotShape","LandContour","Utilities","LotConfig","LandSlope",
#                      "Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle",
#                      "RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond",
#                      "Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
#                      "Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional",
#                      "GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","SaleType",
#                      "SaleCondition"]
#
#for col in train_cat_col_list:
#    train_num.drop(col, axis = 1, inplace = True)

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
#train_cat = train.copy()
#
#train_num_col_list = ["MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd",
#                      "MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
#                      "1stFlrSF","2ndFlrSF","BsmtHalfBath","LowQualFinSF","GrLivArea","BsmtFullBath",
#                      "FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces",
#                      "GarageYrBlt","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch",
#                      "3SsnPorch","ScreenPorch","PoolArea","MiscVal","MoSold","YrSold",
#                      "SalePrice"]
#
#for col in train_num_col_list:
#    train_cat.drop(col, axis = 1, inplace = True)
    
from sklearn.base import BaseEstimator, TransformerMixin
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

train_tr = DataFrameImputer().fit_transform(train)
            
train_tr_2 = pd.get_dummies(train_tr)

#from sklearn.preprocessing import LabelEncoder
#
#le=LabelEncoder()
#
#for col in train_tr.columns:
#    if train_tr[col].dtypes == 'object':
#        val_counts = train_tr[col].value_counts()
#
#for col in train_tr.columns.values:
#       # Encoding only categorical variables
#       if train_tr[col].dtypes=='object':
#           # Using whole data to form an exhaustive list of levels
#           data=train_tr[col].append(train_tr[col])
#           le.fit(data.values)
#           train_tr[col]=le.transform(train_tr[col])
#
#import category_encoders as ce
## Get a new clean dataframe
#train_cat = train.select_dtypes(include=['object']).copy()
#
## Specify the columns to encode then fit and transform
#encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=["MSZoning",
#                                                                 "Street",
#                                                                 "LotShape",
#                                                                 "LandContour",
#                                                                 "Utilities",
#                                                                 "LotConfig",
#                                                                 "LandSlope",
#                                                                 "Neighborhood",
#                                                                 "Condition1",
#                                                                 "Condition2",
#                                                                 "BldgType",
#                                                                 "HouseStyle",
#                                                                 "RoofStyle",
#                                                                 "RoofMatl",
#                                                                 "Exterior1st",
#                                                                 "Exterior2nd",
#                                                                 "MasVnrType",
#                                                                 "ExterQual",
#                                                                 "ExterCond",
#                                                                 "Foundation",
#                                                                 "BsmtQual",
#                                                                 "BsmtCond",
#                                                                 "BsmtExposure",
#                                                                 "BsmtFinType1",
#                                                                 "BsmtFinType2",
#                                                                 "Heating",
#                                                                 "HeatingQC",
#                                                                 "CentralAir",
#                                                                 "Electrical",
#                                                                 "KitchenQual",
#                                                                 "Functional",
#                                                                 "GarageType",
#                                                                 "GarageFinish",
#                                                                 "GarageQual",
#                                                                 "GarageCond",
#                                                                 "PavedDrive",
#                                                                 "SaleType",
#                                                                 "SaleCondition"
#                                                                 ])
#encoder.fit(train_cat, verbose=1)
#
#train_tr = encoder.transform(train_cat)
#
#from sklearn.base import BaseEstimator, TransformerMixin
#
##This will transform the data by selecting the desired attributes, 
##dropping the rest and converting the resulting DataFrame to NumPy array.
#class DataFrameSelector(BaseEstimator, TransformerMixin):
#    def __init__(self, attribute_names):
#        self.attribute_names = attribute_names
#    def fit(self, X, y=None):
#        return self
#    def transform(self, X):
#        return X[self.attribute_names].values
#
#num_attribs = list(train_num)
#cat_attribs = list(train_cat)
#
#num_pipeline = Pipeline([
#        ('selector', DataFrameSelector(num_attribs)),
#        ('imputer', Imputer(strategy="median")),
#        ('std_scaler',StandardScaler()),
#        ])
#    
##cat_pipeline = Pipeline([
##        ('selector', DataFrameSelector(cat_attribs)),
##        ('label_binarizer',MyLabelBinarizer()),
##        ])
#    
#    
##A Full pipeline handling both the numerical and categorical attributes
#
#from sklearn.pipeline import FeatureUnion
#
#full_pipeline = FeatureUnion(transformer_list = [
#        ("num_pipeline", num_pipeline)
#        ])
#    
##     ("cat_pipeline", cat_pipeline),
#        
#train_prepared = full_pipeline.fit_transform(train)
#
#index = [i for i in range(0, len(train_prepared))]
#
#df = pd.DataFrame(data=train_prepared, index= index)
#
#train_Full = train_tr.append(df, ignore_index = True)