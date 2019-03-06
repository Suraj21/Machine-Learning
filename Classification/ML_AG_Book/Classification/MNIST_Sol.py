# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 08:39:46 2019

@author: suraj
"""

from scipy.io import loadmat
mnist = loadmat("D:\GitRepository\Machine-Learning\Classification\ML_AG_Book\Classification\mnist-original.mat\mnist-original")
X = mnist["data"].T
y = mnist["label"][0]

X.shape
y.shape

import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation= "nearest")


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
import numpy as np

#shuffling the trainng set to guarantee that all cross-validation folds will be similar.
#Moreover some learning algorithm 
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train)


sgd_clf.predict([some_digit])

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits = 3, random_state = 42)

#Measuring accuracy using cross validation
#At each iteration the code creates a clone of the classifier, trains that clone on the training folds,
#and makes predictions on the test folds. Then it counts the no of correct prediction and outputs the ratio 
#of correct predictions

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) 
    
#Measuring accuracy using cross val score
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv = 5, scoring="accuracy")

#CLassifier to classifies that every single image is not 5
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()

#Confusion Matrix

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# Confusion Matrix

# performs K-fold cross-validation, but instead returing the evaluation scores, it returns the prediction 
# made on each test fold.

from sklearn.model_selection import cross_val_predict 
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

# precision and recall
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred) # == 4344 / (4344 + 1307)

recall_score(y_train_5, y_train_pred) # == 4344 / (4344 + 1077) detects the percenage of 5s

#F1 score combine the precision and recall into a single metric
# It is the harmonic mean of the precision and recall. Where regular mean treats all values equally 
# the harmonic mean gives much more weight to the low values.
# As a result the classifier will only get a high F1 score if both recall and precision are high.
# F1 = TP/ (TP + (FN + FP)/2) = 2 * (precision * recall)/(precision + recall)

from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

#with threshold equal to zero the result will be same as that of predict() method
y_scores = sgd_clf.decision_function([some_digit])
y_scores
threshold = 0
y_some_digit_pred = (y_scores > threshold)

threshold = 2000000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
# Raising the threshold will decrease the recall
# How to decide the threshold? - For this first get the scores of all instances in the training set using
# the cross_val_predict() functions, but instead of predictig specify that it to return the decision scores

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3, method="decision_function")

# Now with these scores we can compute precision and recall for all possible threshold using the 
# precision_recall_curve() function

from sklearn.metrics import precision_recall_curve
precisions, recalls , thresholdsa = precision_recall_curve(y_train_5, y_scores)




    
    