# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #take all the rows and the all the column except the last one
y = dataset.iloc[:, 3].values #this means take all the rows and the 3rd column only

#Taking Care Of Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = "mean", axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing  import LabelEncoder, OneHotEncoder
LabelEncoder_X = LabelEncoder() #applies label to the categorical data
X[:,0] = LabelEncoder_X.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0]) #converts the applied lable to different no of rows and columns independent of each other
X = oneHotEncoder.fit_transform(X).toarray()
LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling i.e. converting the long rage and unsymmetrical data with in the range of 0-1
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)