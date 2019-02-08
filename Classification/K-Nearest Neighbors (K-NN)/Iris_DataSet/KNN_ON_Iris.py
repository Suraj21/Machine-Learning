# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 22:38:12 2019

@author: suraj
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cross_validation import train_test_split

# iris = datasets.load_iris()
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv("D:\GitRepository\Machine-Learning\Classification\K-Nearest Neighbors (K-NN)\Iris_DataSet\Iris.csv")

#X = iris.data
# df = pd.DataFrame(X)
# df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df.iloc[:,0:4]
y = df.iloc[:,4]
df.head()

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
accuracy_score(y_test, pred)


# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
# KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, metric=’minkowski’, p=2, metric_params=None)
# ‘n_neighbors‘ are the number of neighbors that will vote for the class of the target point;
# default number is 5. An odd number is preferred to avoid any tie.
# ‘weights‘ parameter has two choices: ‘uniform‘ and ‘distance‘. For the ‘uniform‘ weight, 
# each of the k neighbors has equal vote whatever its distance from the target point.
# If the weight is ‘distance‘ then voting weightage or importance varies by inverse of distance; 
# those points who are nearest to the target point have greater influence than those who are farther away.
# Parameter ‘algorithm‘ is for selecting the indexing data structure that will be used for speeding up 
# neighborhood search; value of ‘auto‘ leaves it to algorithm to make the best choice among the three.
# Three algorithms are:  brute, kd_tree and ball_tree 
# Parameter ‘leaf_size‘ is the size of leaf in kd_tree or ball_tree. 
# Larger the size, greater the speed of initial indexing structure formation but at the cost of delay
# in classification of target point.
# Parameter ‘metric‘ decides how distances are calculated in space. One familiar way is euclidean distance
# but then in some cases other measures of distances such as Manhattan distance are also used. 
# A general formulation of distance metric is ‘minkowski’ distance. When parameter ‘p‘ is 2, 
# it is the same as euclidean distance and when parameter ‘p‘ is 1, it is Manhattan distance.
# Parameter ‘metric‘ decides how distances are calculated in space. One familiar way is euclidean distance 
# but then in some cases other measures of distances such as Manhattan distance are also used. 
# A general formulation of distance metric is ‘minkowski’ distance. When parameter ‘p‘ is 2, 
# it is the same as euclidean distance and when parameter ‘p‘ is 1, it is Manhattan distance.

#CV = 10 means that we are performing 10 folds cross validation and our scoring metrix should be accuracy

# HERE FOR EACH NEIGHBOUR LIST WE HAVE WE ARE FINDING THE MINIMUN CROSS VALIDATION SCRORE, SO THAT WE CAN USE THAT K
# FOR CREATING BETTER MODEL
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy') 
    cv_scores.append(scores.mean())


# Finally, we plot the misclassification error versus K.
    
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print(optimal_k)

# plot misclassification error vs k
import matplotlib.pyplot as plt
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()



