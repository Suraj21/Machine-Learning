# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:50:06 2019

@author: suraj
"""

#Grid Search
# Answers - How do I know which parameter to select when I make machine learning model, what is the optimal
# value for the hyperparamter.
#1. How to select the ML Model?
#Ans -  Look at the problem whether it has dependent variable or not 
# If it has no dependent variable then it is a clustering problem (UnSupervised) else it is a supervised 
# problem. Then Look for output whether it has continuous target variable (Regression)
# or categorical target variable (Classification).
# 2. Whether the problem is linear problem or a Non-Linear Model i.e. Linearly seperable or not.


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
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

# Applying Grid Search to find the best model and the best parameter
# To define the parameters look for the paramters what the SVC takes like 'C' as Penaltiy parameter and 
# what all kernal value it takes, like linear or rbf (Gaussian kernel) (non linear kernal)
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000], 'kernel':['linear']},
               {'C':[1,10,100,1000], 'kernel':['rbf'],'gamma':[0.5, 0.1, 0.01, 0.001, 0.0001]}
               ]

# Classifier is which kernal model we are going to use like above we have used support vector classifier
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters,
                           scoring='accuracy', cv=10, n_jobs= -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
best_estimatro = grid_search.best_estimator_


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()