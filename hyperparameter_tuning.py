import xgboost
import numpy as np 
import pandas as pd 
from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
X = pd.read_csv("final_X_train.csv")
Y = pd.read_csv("final_y_train.csv")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
regressor=xgboost.XGBRegressor(booster = 'gbtree')
base_score=[0.25,0.5,0.6,0.7]
n_estimators = [100, 500,600,700,900]
max_depth = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
learning_rate=[0.13,0.12,0.15,0.14,0.16]
min_child_weight=[1,2,3,4]



hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'base_score':base_score
    }


random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)

random_cv.fit(X_train,y_train)
print(random_cv.best_estimator_)