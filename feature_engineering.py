import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
pd.pandas.set_option('display.max_columns',None)

dataset = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
#Numerical null values
#Categorical Null values 
features_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes=='O']
 #its null and is an object
features_nan_test = [feature for feature in test_data.columns if test_data[feature].isnull().sum()>1 and test_data[feature].dtypes =='O']

#for feature in features_nan:
    #print("{}: {}% missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))  #percentage of missing values
#print("In test_data")
#for feature in features_nan_test:
    #print("{}: {}% missing values".format(feature,np.round(test_data[feature].isnull().mean(),4)))


def replace_cat_feature(dataset,features_nan):
    data=dataset.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

dataset=replace_cat_feature(dataset,features_nan)
test_data = replace_cat_feature(test_data,features_nan_test)

#print(dataset.head())
#print(test_data.head())

#print(dataset[features_nan].isnull().sum())
#print(test_data[features_nan_test].isnull().sum())


numerical_with_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']
numerical_with_nan_test = [feature for feature in test_data.columns if test_data[feature].isnull().sum()>1 and test_data[feature].dtypes!='O']


#for feature in numerical_with_nan:
    #print("{}: {}% missing value".format(feature,np.around(dataset[feature].isnull().mean(),4)))


#for feature in numerical_with_nan_test:
    #print("{}: {}% missing value".format(feature,np.around(test_data[feature].isnull().mean(),4)))


for feature in numerical_with_nan:
    ## We will replace by using median since there are outliers
    median_value=dataset[feature].median()
    
    ## create a new feature to capture nan values
    dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0) 
    dataset[feature].fillna(median_value,inplace=True)
    
#print(dataset[numerical_with_nan].isnull().sum())



for feature in numerical_with_nan_test:
    ## We will replace by using median since there are outliers
    median_value=test_data[feature].median()
    
    ## create a new feature to capture nan values
    test_data[feature+'nan']=np.where(test_data[feature].isnull(),1,0) 
    test_data[feature].fillna(median_value,inplace=True)
    

#print(test_data[numerical_with_nan_test].isnull().sum())

# Non Gaussian variables

num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
num_features_test=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']

for feature in num_features:
    dataset[feature]=np.log(dataset[feature])

for feature in num_features_test:
    test_data[feature]=np.log(test_data[feature])


#print(dataset[num_features].head())
#print(test_data[num_features_test].head())

categorical_features=[feature for feature in dataset.columns if dataset[feature].dtype=='O']
#print(categorical_features)


for feature in categorical_features:
    temp=dataset.groupby(feature)['Id'].count()/len(dataset)
    temp_df=temp[temp>0.01].index
    dataset[feature]=np.where(dataset[feature].isin(temp_df),dataset[feature],'Rare_var')

for feature in categorical_features:
    temp=test_data.groupby(feature)['Id'].count()/len(test_data)
    temp_df=temp[temp>0.01].index
    test_data[feature]=np.where(test_data[feature].isin(temp_df),test_data[feature],'Rare_var')




for feature in categorical_features:
    labels_ordered=dataset.groupby([feature])['Id'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    print(labels_ordered)
    dataset[feature]=dataset[feature].map(labels_ordered)
    test_data[feature] = test_data[feature].map(labels_ordered)

print(dataset[categorical_features].head())
print(test_data[categorical_features].head())


feature_scale=[feature for feature in dataset.columns if feature not in ['Id','SalePrice']]
feature_scale_test=[feature for feature in test_data.columns if feature not in ['Id']]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(dataset[feature_scale])


scaler.transform(dataset[feature_scale])


data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dataset[feature_scale]), columns=feature_scale)],
                    axis=1)

data.to_csv('feature_scaled_train.csv',index=False)



scaler.fit(test_data[feature_scale_test])


scaler.transform(test_data[feature_scale_test])


data_test = pd.concat([test_data[['Id']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(test_data[feature_scale_test]), columns=feature_scale_test)],
                    axis=1)

data_test.to_csv('feature_scaled_test.csv',index=False)