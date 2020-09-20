import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

#display all columns of the dataframe(rows also can be shown)
pd.pandas.set_option('display.max_columns',None)

dataset = pd.read_csv('train.csv')
print(dataset.shape)
#Missing values
features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]
for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean(), 4),  ' % missing values')


for feature in features_with_na:
	data = dataset.copy()

# let's make a variable that indicates 1 if the observation is null or zero if not null
	data[feature] = np.where(data[feature].isnull(), 1, 0)

# let's calculate the mean SalePrice where the information is missing or present
	data.groupby(feature)['SalePrice'].median().plot.bar()
	plt.title(feature)
	plt.show()

#NUMERICAL VALUES
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O'] #'0' means object
#date time type values
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
print(year_feature)

for feature in year_feature:
    print(feature, dataset[feature].unique())


dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")

for feature in year_feature:
    if feature!='YrSold':
        data=dataset.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


#discrete features
discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']] #if the number is less then 25 and they are not temporal variables or id

print("Discrete Variables Count: {}".format(len(discrete_feature)))




for feature in discrete_feature:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()



#continuous features
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))
#plot histograms
for feature in continuous_feature:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


#logarithmic transformation
for feature in continuous_feature:

	data = dataset.copy()
	if 0 in data[feature].unique(): #log 0 is not defined
		pass
	else:
		data[feature] = np.log(data[feature])
		data['SalePrice'] = np.log(data['SalePrice'])
		plt.scatter(data[feature],data['SalePrice'])
		plt.xlabel(feature)
		plt.ylabel('SalesPrice')
		plt.title(feature)
		plt.show()

#Outliers box plot
for feature in continuous_feature:

	data = dataset.copy()
	if 0 in data[feature].unique(): #log 0 is not defined
		pass
	else:
		data[feature] = np.log(data[feature])
		dataset.boxplot(column=feature)
		plt.ylabel(feature)
		plt.title(feature)
		plt.show()

#Categorical variables
categorical_features=[feature for feature in dataset.columns if data[feature].dtypes=='O']


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(dataset[feature].unique())))

for feature in categorical_features:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()