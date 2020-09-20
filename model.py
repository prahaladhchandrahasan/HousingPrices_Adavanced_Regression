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


nan = None
regressor=XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.15, max_delta_step=0, max_depth=4,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)








regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)
print("rms train  error:",mean_squared_error(y_train, y_pred_train,squared = False))
print("rms test  error:",mean_squared_error(y_test, y_pred,squared = False))


test = pd.read_csv('final_test_data.csv')

predictions = regressor.predict(test)
pred=pd.DataFrame(np.exp(predictions))
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('my_submission.csv',index=False)