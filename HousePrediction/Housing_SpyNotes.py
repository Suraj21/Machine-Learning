# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 20:08:40 2019

@author: suraj
"""

import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "dataset/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
print(HOUSING_URL)

def fetch_housing_data(housing_url=HOUSING_URL, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    

def load_housing_data(housing_path="D:\GitRepository\MLWithSKLearnAndTFBook\handson-ml\datasets\housing"):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()

housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing_description = housing.describe()

#%matplotlib inline
housing.hist(bins=50, figsize=(20,15))
plt.show()

#  this method will generate the different test and train sets every time it is executed
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

import hashlib
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # adds and 'index' column to the housig set
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"]*1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
housing["income_cat"].value_counts() / len(housing)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4)

housing["population"]

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7),
            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

corr_matrix =housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

housing.plot(kind="scatter",x="median_income", y="median_house_value", alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

housing["rooms_per_household"] 
housing["bedrooms_per_room"]
housing["population_per_household"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False) #shows bedrooms_per_room  is much more correlated with median house value than the total no of rooms or bedrooms

#preparing the data for machine learning
housing = strat_train_set.drop("medain_house_value",axis=1) #drop creates the copy of the data and does not affect the  strat_train_set
housing_labels = strat_train_set["median_house_value"].copy()

#Data Cleaning (Removing the columns with the missing values Or filling with the median)
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

#using Imputer
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num) #imputer has computed the median of each attributes and stored the result in it's statistics_ instance variable.
#it is safer to apply the imputer to all the numerical attributes as we can't be sure which column can't have missing values
imputer.statistics_ #trained imputer which is based on the housing median price
housing_num.median().values
X = imputer.transform(housing_num) #The output is the trained numpy array

#converting it the pandas DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# Handling text and categorical attributes
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat) #fit_transform expects 2D array but housing_cat_encoded is 1D array and needs to be reshaped
print(housing_cat_encoded)

print(encoder.classes_)

from sklearn.preprocesing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot # this will output in a sparse matrix to convert it to a dense matrix use to array
housing_cat_1hot.toarray() # converting it to dense matrix

# To aply both transformation i.e. from text categories to integer categories, then from 
# integer categories to one-hot vectors in one shot we can use LableBinarizer class

from sklearn.preprocessing import LabelBinarizer
labelBinarizer_encoder = LabelBinarizer()
housing_cat_1hot = labelBinarizer_encoder.fit_transform(housing_cat)
housing_cat_1hot # this will output in the numpy dense array by default

#Custom Transformers

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no * args or ** kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self # nothinhg else to do
    
    def transform(self, X, y=None):
        rooms_per_household = X[:,rooms_ix]/X[:, household_ix]
        population_per_household = X[: population_ix] / X[:, rooms_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X:, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)

#Transform Pipeline

# There are many transformation steps that needs to be executed in the right order and for 
# this Scikit-Learn provides the pipeline class to help with such sequences of transformation

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder',CombinedAttributesAdder()),
        ('std_scaler',StandardScaler()),
        ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

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
    
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler',StandardScaler()),
        ])

from sklearn.preprocessing import LabelBinarizer
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer',LabelBinarizer()),
        ])
    
#A Full pipeline handling both the numerical and categorical attributes

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list = [
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
        ])
        
housing_prepared = full_pipeline.fit_transform(housing)

#Training and evaluating on the Training set
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
lin_reg.predict(some_data_prepared)

#Looking for RMSE error 
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_mse = np.sqrt(lin_mse)


