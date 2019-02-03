# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 07:49:14 2019

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import  preprocessing 


from sklearn import datasets
iris = datasets.load_iris()
I_Data = iris.data

from sklearn import datasets
difits = datasets.load_digits()

    
dataset = pd.read_csv('Titanic_train.csv')


dataset = dataset.drop(['Cabin'], axis=1) 
dataset = dataset.dropna()

""" Data preprocessing function"""
def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['Name','Ticket'],axis=1)
    return processed_df

processed_df = preprocess_titanic_df(dataset)

X = processed_df.loc[:, processed_df.columns != 'Survived'].values # fetching all the columns except the surviced as considering that as the target 
y = processed_df.iloc[:, 1].values # fetching only the survived columns 

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


from sklearn.tree import DecisionTreeClassifier
clf_DT_Default = DecisionTreeClassifier(random_state=999)
clf_DT_Default.fit(X_train, y_train)
pred_Default = clf_DT_Default.predict(X_test)
print(accuracy_score(y_test, pred_Default))


clf_dt = DecisionTreeClassifier(criterion = "entropy",max_depth=10,min_samples_split = 5,random_state =999) # Define model
clf_dt.fit (X_train, y_train) # Fit model with your data
predictions = clf_dt.predict(X_test) # Preditions on test data
#from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)

