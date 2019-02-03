# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 18:12:03 2018

@author: DH384961
"""


import numpy as np # libraries for array operations
import pandas as pd # for data handling
from sklearn import  preprocessing #data sampling,model and preprocessing 

path ="E:/DHANANJAY_R_WIPRO/Class/Data/Titanic_train.csv"
#g_path = "./"
titanic_df = pd.read_csv(path,encoding="utf8" ) # read Data

titanic_df.head() # print few data


""" Data exploration and processing"""
titanic_df['Survived'].mean()


titanic_df.groupby('Pclass').mean()

class_sex_grouping = titanic_df.groupby(['Pclass','Sex']).mean()
class_sex_grouping

class_sex_grouping['Survived'].plot.bar()       


group_by_age = pd.cut(titanic_df["Age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['Survived'].plot.bar()     

titanic_df.count()

titanic_df = titanic_df.drop(['Cabin'], axis=1)   

 
titanic_df = titanic_df.dropna()    

titanic_df.count()

""" Data preprocessing function"""
def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['PassengerId','Name','Ticket'],axis=1)
    return processed_df
    

processed_df = preprocess_titanic_df(titanic_df)

# SVM  Modeling  using SKlearn
X = processed_df.drop(['Survived'], axis=1)# Features dataset
y = processed_df['Survived'] # Target variable

#Train Test split
from sklearn.cross_validation import train_test_split as tt_split
X_train, X_test, y_train, y_test = tt_split(X,y,test_size=0.2,random_state = 100)

from sklearn.metrics import accuracy_score as ac
from sklearn.ensemble import RandomForestClassifier as RFC
RF_default = RFC(random_state =10)
RF_default.fit(X_train, y_train)
pred_dfault = RF_default.predict(X_test)
ac(y_test,pred_dfault)

RF_model1 = RFC(n_estimators = 7,criterion ="entropy",
                max_features = 2, max_depth= 9 , random_state =999)

RF_model1.fit(X_train,y_train)
pred_model1 = RF_model1.predict(X_test)
pred_model1_train = RF_model1.predict(X_train)
ac(y_train,pred_model1_train)
ac(y_test,pred_model1)


from sklearn.model_selection import GridSearchCV
params = {
            'n_estimators': [10,15,20,50], 'criterion': ["entropy","gini"],
              'max_features' : [2,3,7],     'max_depth' : [5,10,15]
        } 
       
grid_RF = GridSearchCV(estimator=RF_default, param_grid= params,cv =3)
grid_RF.fit(X_train,y_train)

grid_RF.best_estimator_

final_model = grid_RF.best_estimator_
final_model.fit(X_train,y_train)
final_pred = final_model.predict(X_test)
ac(y_test,final_pred)






 