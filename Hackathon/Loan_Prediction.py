# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:09:46 2019

@author: suraj
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('D:\ML_DataSets_Open\Loan_Prediction\Train.csv')
test = pd.read_csv('D:\ML_DataSets_Open\Loan_Prediction\Test.csv')
train_X = train.iloc[:,1:12]
loan_id = test.iloc[:,:1]
test = test.iloc[:,1:12]
train_y = train.iloc[:,12]

dataset = pd.concat([train_X,test])

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

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
    
dataset = DataFrameImputer().fit_transform(dataset)

dataset.drop("Dependents", axis=1, inplace = True)
dataset.drop("ApplicantIncome", axis=1, inplace = True)
dataset.drop("Loan_Amount_Term", axis=1, inplace = True)
dataset.drop("Gender", axis=1, inplace = True)
#dataset.drop("Married", axis=1, inplace = True)
#dataset.drop("Self_Employed", axis=1, inplace = True)
dataset.drop("CoapplicantIncome", axis=1, inplace = True)
dataset.drop("LoanAmount", axis=1, inplace = True)
dataset.drop("Education", axis=1, inplace = True)
dataset.drop("Married", axis=1, inplace = True)
dataset.drop("Self_Employed", axis=1, inplace = True)
dataset['Credit_History'] = dataset['Credit_History'].replace(0.835920177383592, value = 1)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()

dataset.iloc[:, 0] = labelencoder_1.fit_transform(dataset.iloc[:, 0])
dataset.iloc[:, 1] = labelencoder_1.fit_transform(dataset.iloc[:, 1])
dataset.iloc[:, 3] = labelencoder_1.fit_transform(dataset.iloc[:, 3])
dataset.iloc[:, 4] = labelencoder_1.fit_transform(dataset.iloc[:, 4])
dataset.iloc[:, 5] = labelencoder_1.fit_transform(dataset.iloc[:, 5])
dataset.iloc[:, 9] = labelencoder_1.fit_transform(dataset.iloc[:, 9])

onehotencoder_1 = OneHotEncoder()
dataset = onehotencoder_1.fit_transform(dataset).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset = sc.fit_transform(dataset)
X_train = dataset[:614,:]
X_test = dataset[614:981,:]

# Fitting XGBoost to the training set
from xgboost import XGBClassifier
xgclassifier = XGBClassifier(max_depth=5,
                           min_child_weight=5,
                           learning_rate=0.1,
                           n_estimators=1000,
                           silent=True,
                           objective='binary:logistic',
                           gamma=0,
                           max_delta_step=0,
                           subsample=1,
                           colsample_bytree=1,
                           colsample_bylevel=1,
                           reg_alpha=0,
                           reg_lambda=0,
                           scale_pos_weight=1,
                           seed=1,
                           missing=None)
xgclassifier.fit(X_train, train_y)

# Predicting the Test set results
y_predxg = xgclassifier.predict(X_test)
loan_id['Loan_Status'] = y_predxg
prediction = pd.DataFrame(loan_id, columns=['Loan_ID','Loan_Status']).to_csv('D:\ML_DataSets_Open\Loan_Prediction\Sample_Submission.csv', index=False)

#Applying K-fold cross validation
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= xgclassifier, X = X_train, y = train_y, cv=10)
accuracies.mean()
accuracies.std()


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
dtcclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtcclassifier.fit(X_train, train_y)

#Applying K-fold cross validation
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= dtcclassifier, X = X_train, y = train_y, cv=10)
accuracies.mean()
accuracies.std()

# Predicting the Test set results
y_preddtc = dtcclassifier.predict(X_test)
prediction = pd.DataFrame(y_preddtc, columns=['Loan_Status']).to_csv('D:\ML_DataSets_Open\Loan_Prediction\Sample_Submission.csv')

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
svcclassifier = SVC(kernel = 'rbf', random_state = 0)
svcclassifier.fit(X_train, train_y)

# Predicting the Test set results
y_predsvc = svcclassifier.predict(X_test)

#Applying K-fold cross validation
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= svcclassifier, X = X_train, y = train_y, cv=10)
accuracies.mean()
accuracies.std()

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
neighclassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
neighclassifier.fit(X_train, train_y)

# Predicting the Test set results
y_predknn = neighclassifier.predict(X_test)

#Applying K-fold cross validation
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= neighclassifier, X = X_train, y = train_y, cv=10)
accuracies.mean()
accuracies.std()

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
lgclassifier = LogisticRegression(random_state = 0)
lgclassifier.fit(X_train, train_y)

# Predicting the Test set results
y_predlg = lgclassifier.predict(X_test)

#Applying K-fold cross validation
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= lgclassifier, X = X_train, y = train_y, cv=10)
accuracies.mean()
accuracies.std()

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rfclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfclassifier.fit(X_train, train_y)

# Predicting the Test set results
y_predrf = rfclassifier.predict(X_test)

#Applying K-fold cross validation
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= rfclassifier, X = X_train, y = train_y, cv=10)
accuracies.mean()
accuracies.std()

# Fitting SVM to the Training set
from sklearn.svm import SVC
svmclassifier = SVC(kernel = 'sigmoid', random_state = 0)
svmclassifier.fit(X_train, train_y)

# Predicting the Test set results
y_predsvm = svmclassifier.predict(X_test)

#Applying K-fold cross validation
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= svmclassifier, X = X_train, y = train_y, cv=10)
accuracies.mean()
accuracies.std()

from sklearn.svm import SVC
svmrbfclassifier = SVC(kernel = 'rbf', random_state = 0)
svmrbfclassifier.fit(X_train, train_y)

# Predicting the Test set results
y_predsvmrbf = svmrbfclassifier.predict(X_test)


#Applying K-fold cross validation
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= svmrbfclassifier, X = X_train, y = train_y, cv=10)
accuracies.mean()
accuracies.std()

#Appying Grid search
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000], 'kernel':['linear']},
              {'C':[1,10,100,1000], 'kernel':['rbf'],'gamma':[0.5, 0.1, 0.01, 0.001, 0.0001]}
             ]

# Classifier is which kernal model we are going to use like above we have used support vector classifier
grid_search = GridSearchCV(estimator = svmrbfclassifier, param_grid = parameters,
                           scoring='accuracy', cv=10)

grid_search = grid_search.fit(X_train, train_y)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
best_estimatro = grid_search.best_estimator_


#Applying K-fold cross validation
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= xgclassifier, X = X_train, y = train_y, cv=10)
accuracies.mean()
accuracies.std()





