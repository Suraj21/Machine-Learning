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
dataset["Income"] = dataset["CoapplicantIncome"] + dataset["ApplicantIncome"]

dataset.drop("Dependents", axis=1, inplace = True)
dataset.drop("ApplicantIncome", axis=1, inplace = True)
dataset.drop("Loan_Amount_Term", axis=1, inplace = True)
#dataset.drop("Gender", axis=1, inplace = True)
#dataset.drop("Married", axis=1, inplace = True)
dataset.drop("Self_Employed", axis=1, inplace = True)
dataset.drop("CoapplicantIncome", axis=1, inplace = True)
#ataset.drop("LoanAmount", axis=1, inplace = True)
#dataset.drop("Education", axis=1, inplace = True)
dataset['Credit_History'] = dataset['Credit_History'].replace(0.835920177383592, value = 1)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()

dataset.iloc[:, 0] = labelencoder_1.fit_transform(dataset.iloc[:, 0])
dataset.iloc[:, 1] = labelencoder_1.fit_transform(dataset.iloc[:, 1])
dataset.iloc[:, 2] = labelencoder_1.fit_transform(dataset.iloc[:, 2])
dataset.iloc[:, 5] = labelencoder_1.fit_transform(dataset.iloc[:, 5])
#dataset.iloc[:, 4] = labelencoder_1.fit_transform(dataset.iloc[:, 4])
#dataset.iloc[:, 10] = labelencoder_1.fit_transform(dataset.iloc[:, 10])
#dataset.iloc[:, 9] = labelencoder_1.fit_transform(dataset.iloc[:, 9])

predictors = [x for x in train.columns if x not in ["Loan_ID", "Dependents","ApplicantIncome",
                                                    "Loan_Status","Self_Employed","CoapplicantIncome",
                                                    ]]
predictors.append('Income')

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
#xgclassifier = XGBClassifier(max_depth=2,
#                           min_child_weight=5,
#                           learning_rate=0.1,
#                           n_estimators=25,
#                           silent=True,
#                           objective='binary:logistic',
#                           gamma=0,
#                           max_delta_step=0,
#                           subsample=1,
#                           colsample_bytree=1,
#                           colsample_bylevel=1,
#                           reg_alpha=0,
#                           reg_lambda=1.2,
#                           scale_pos_weight=1,
#                           seed=1,
#                           missing=None)
xgclassifier = XGBClassifier(n_estimators=100, min_child_weight = 3,reg_lambda = 1.2)
xgclassifier.fit(X_train, train_y)

train_y = train_y.replace('Y',1)
train_y = train_y.replace('N',0)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, train_y, test_size=0.25)

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(x_train, y_train)
y_pred = gbm0.predict(x_test)

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

#learning_rate
#learning rate shrinks the contribution of each tree by learning_rate.
learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
train_results = []
test_results = []
for eta in learning_rates:
   model = GradientBoostingClassifier(learning_rate=eta)
   model.fit(x_train, y_train)
   train_pred = model.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = model.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(learning_rates, train_results, 'b', label="Train AUC")
line2, = plt.plot(learning_rates, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('learning rate')
plt.show()
# As per the graph 0.1 learning rate is the optimal rate

#n_estimators
#n_estimators represents the number of trees in the forest. Usually the higher the number of 
#trees the better to learn the data. However, adding a lot of trees can slow down the 
#training process considerably, therefore we do a parameter search to find the sweet spot.

n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
   model = GradientBoostingClassifier(n_estimators=estimator)
   model.fit(x_train, y_train)
   train_pred = model.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = model.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()
#As per the graph 25 is the optimal value for n_estimators

#max_depth. 
# This indicates how deep the built tree can be. The deeper the tree,
# the more splits it has and it captures more information about how the data. 
# We fit a decision tree with depths ranging from 1 to 32 and plot the training 
# and test errors.

max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
   model = GradientBoostingClassifier(max_depth=max_depth)
   model.fit(x_train, y_train)
   train_pred = model.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = model.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()

# min_samples_split represents the minimum number of samples required to split an internal node.
# This can vary between considering at least one sample at each node to considering all 
# of the samples at each node. When we increase this parameter, the tree becomes more 
# constrained as it has to consider more samples at each node. Here we will vary 
# the parameter from 10% to 100% of the samples

min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   model = GradientBoostingClassifier(min_samples_split=min_samples_split)
   model.fit(x_train, y_train)
   train_pred = model.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = model.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()
#We can clearly see that when we require all of the samples at each node, 
# the model cannot learn enough about the data. This is an underfitting case.


#min_samples_leaf is The minimum number of samples required to be at a leaf node.
# This similar to min_samples_splits, however, this describe the minimum number of 
# samples of samples at the leafs.

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
   model = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf)
   model.fit(x_train, y_train)
   train_pred = model.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = model.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leafs, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_leafs, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.show()
# Increasing this value can cause underfitting.

#max_features = ‘sqrt’ : Its a general thumb-rule to start with square root.
#max_features represents the number of features to consider when looking for the best split.

max_features = list(range(1,train.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
   model = GradientBoostingClassifier(max_features=max_feature)
   model.fit(x_train, y_train)
   train_pred = model.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = model.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_features, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max features')
plt.show()

#subsample = 0.8 : This is a commonly used used start value
#feat_imp = pd.Series(gbm0.feature_importances_, predictors).sort_values(ascending=False)
#feat_imp.plot(kind='bar', title='Feature Importances')
#plt.ylabel('Feature Importance Score')

import matplotlib.pylab as plt 
feat_imp = pd.Series(xgclassifier.get_booster().get_score(importance_type='gain')).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

# Predicting the Test set results
y_predxg = xgclassifier.predict(X_test)
loan_id['Loan_Status'] = y_predxg
prediction = pd.DataFrame(loan_id, columns=['Loan_ID','Loan_Status']).to_csv('D:\ML_DataSets_Open\Loan_Prediction\Sample_Submission.csv', index=False)

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

xgb_model = XGBClassifier(max_depth=1,n_estimators=50, min_child_weight = 2,reg_lambda = 1.15)
optimization_dict = {'max_depth': [1,2,3,5],
                     'n_estimators': [50,75,100,125],
                     'min_child_weight':[2,3,4,5],
                     'reg_lambda':[1.15,1.2,1.25,1.27,1.3],
                     
                    }

model = GridSearchCV(xgb_model, optimization_dict, scoring='accuracy', verbose=1,cv=3)
model.fit(X_train, train_y)
print(model.best_score_)
print(model.best_params_)

xgb_model.fit(X_train, train_y)
test_pred_clf = xgb_model.predict(X_test)
loan_id['Loan_Status'] = test_pred_clf
prediction = pd.DataFrame(loan_id, columns=['Loan_ID','Loan_Status']).to_csv('D:\ML_DataSets_Open\Loan_Prediction\Sample_Submission.csv', index=False)


#Applying K-fold cross validation
from sklearn.model_selection import  cross_val_score
accuracies = cross_val_score(estimator= xgclassifier, X = X_train, y = train_y, cv=10)
print(accuracies.mean())
print(accuracies.std())


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





