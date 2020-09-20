import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

## for feature slection

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

dataset=pd.read_csv('feature_scaled_train.csv')

y_train=dataset[['SalePrice']]

X_train=dataset.drop(['Id','SalePrice'],axis=1)

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_train, y_train)


print(feature_sel_model.get_support())


selected_feat = X_train.columns[(feature_sel_model.get_support())]

# let's print some stats
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))



X_train=X_train[selected_feat]

X_train.to_csv('final_X_train.csv',index=False)
y_train.to_csv('final_y_train.csv',index=False)


dataset_test=pd.read_csv('feature_scaled_test.csv')
test_data = dataset_test.drop(['Id'],axis=1)
test_data=test_data[selected_feat]
test_data.to_csv('final_test_data.csv',index=False)

