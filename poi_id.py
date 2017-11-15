#!/usr/bin/python


import pickle
import numpy as np
import pandas as pd

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

"""
	Section #1 

	-Data set information , outliers and cleanning 
	
	First of all lest find out some data about the loaded data set like total number of 
	data point , allocation of target classes (POI vs NON POI), original number of features
	missing values, features containing missing values and outliers (detection and management)
	To accomplish that i will use, as recommended, pandas dataframe.
	Additionaly, i will drop all non-finantial features from the dataset,

"""

### Load the pickle file
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Copy the loaded data to a new structure to manipulate it
my_dataset = data_dict

### Convert the data set to pandas using float64 as dtype and employee names as index
my_dataset = pd.DataFrame.from_dict(my_dataset, orient='index', dtype='float64')

### Obtain total number of data points, poi and non poi
print "Data set information"
print "Total number of datapoints : {}".format(len(my_dataset.index))
print "Allocation of poi / non poi : \n{}".format(my_dataset['poi'].value_counts())
print "Total number of features : {}".format(len(my_dataset.columns))

### Lets obtain the list of original features that came with the dataset
original_features = my_dataset.columns.values

### Lets check for 'Missing data' in the features . Build two list for if is required
### latter 
has_missing_list = []
has_not_missing_list = []

for feat in original_features:
	has_missing = my_dataset[feat].isnull().values
	if has_missing.any() == True:
		has_missing_list.append(feat)
	else:
		has_not_missing_list.append(feat)


### Lets print the lists
print "\n\nData set information"
print "Features with missing values: {}".format(has_missing_list)
print "Features with no missing values: {}".format(has_not_missing_list)

### Drop email related features
mail_features = ['to_messages','shared_receipt_with_poi','from_messages','from_this_person_to_poi','email_address','from_poi_to_this_person']
my_dataset = my_dataset.drop(mail_features, axis=1)
### Print and allocate a list with the remaining features
initial_features = my_dataset.columns.values
print "\n\nFeatures remaining after cleaning mail: {}".format(my_dataset.columns.values)

### Drop 'TOTAL' index and any data point with all 'NaN'
my_dataset = my_dataset.drop(['TOTAL'])
my_dataset = my_dataset.dropna(how='all')

### Identify features with outliers. Will be treated on features engineering
print "\n\nOutliers identification"
print "Features . Distance in max - (2*std)"
print my_dataset.max() - (my_dataset.mean() + (2 * my_dataset.std()))

### Fill NaN with 0
my_dataset = my_dataset.fillna(0)

""" 
	Section #2

	-Feature selection engineering 
	First i will train and evaluate a RandomForestClassifier with the features i end up
	I will use StratifiedShuffleSplit 
	...............


"""

### First convert back the dataframe to extract features and labels
# my_dataset = my_dataset.to_dict(orient='index')

### First we will train a model without new features . We'll use the "original_features" list
### which cointains what features remain after cleaning mail features. Lets print the list
print "\n\n This is the list of features for the first train and validation: {}".format(initial_features)

### Extract features and labels from dataset for local testing

labels_series = my_dataset.loc[:, 'poi']
labels = labels_series.values
labels = labels.astype(np.int64)

### Now the features, we ahould 'index' the slicing with a list of all the features BUT poi
features_columns = []
for col in my_dataset.columns:
    if col != 'poi':
        features_columns.append(col)

### pandas 'as_matrix' helps convert pandas Series to ndarrays based on selected columns
features = my_dataset.as_matrix(columns=features_columns)

### Lets check the results. labels should be a 1D array and features a ND array 
print "labels has this shape: {}".format(labels.shape)
print "features has this shape: {}".format(features.shape)


### Lets build the model for feature exploration
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, stratify=labels, random_state=42)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(features_train, labels_train)
feat_importance = forest.feature_importances_

my_dataset_no_poi = my_dataset.drop('poi', axis=1)

feat_names = my_dataset_no_poi.columns.values
test = pd.Series(feat_importance,index=feat_names)
print test[test > 0.10]

### **************** Lets move to build a different feature list creating cash_from_stock **********
### labels can be reused, features will be redefined
### Firts copy the dataset 
my_magic_dataset = my_dataset

### Create new feature
# my_magic_dataset['cash_from_stock'] = my_magic_dataset['exercised_stock_options'] + my_magic_dataset['restricted_stock']
# my_magic_dataset['restricted_ratio'] = my_magic_dataset['exercised_stock_options'] / my_magic_dataset['restricted_stock']
# my_magic_dataset['cash_from_stock'] = my_magic_dataset['exercised_stock_options'] + my_magic_dataset['restricted_stock']

water_mark = my_magic_dataset['exercised_stock_options'].quantile(.80)
my_magic_dataset['high_exercised_percentile'] = my_magic_dataset[my_magic_dataset['exercised_stock_options'] > water_mark]

### Drop the non required features
# non_required_features = ['exercised_stock_options','restricted_stock','total_stock_value']
my_magic_dataset = my_magic_dataset.drop(non_required_features, axis=1)

my_magic_dataset_no_poi = my_magic_dataset.drop('poi', axis=1)
magic_feat = my_magic_dataset_no_poi.columns.values
my_magic_features = my_magic_dataset_no_poi.as_matrix(columns=magic_feat)

print "labels has this shape: {}".format(labels.shape)
print "features has this shape: {}".format(my_magic_features.shape)

magic_features_train, magic_features_test, labels_train, labels_test = train_test_split(my_magic_features, labels, stratify=labels, random_state=42)

magic_forest = RandomForestClassifier(n_estimators=100, random_state=0)
magic_forest.fit(magic_features_train ,labels_train)
magic_feat_importance = magic_forest.feature_importances_

# my_dataset_no_poi = my_dataset.drop('poi', axis=1)

magic_feat_names = my_magic_dataset_no_poi.columns.values
magic_test = pd.Series(magic_feat_importance,index=magic_feat_names)
print "Magic features best score: {}".format(forest.best_score_)
print magic_test[magic_test > 0.10]

### Lets build 

### Lets build our StratifiedShuffleSplit for the dataset
### Build max_depth and n_estimators for the param_grid
# tree_param_grid = {'max_depth': [1, 2, 4, 8], 'n_estimators': [1,5,10,20,50,100]}

### Lets build our StratifiedShuffleSplit for the dataset
#tree_cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

### Build GridSearchCV
# tree = RandomForestClassifier(max_features=None, random_state=0), param_grid=tree_param_grid, cv=tree_cv)

### Fit it
#tree_grid.fit(features, labels)

#print tree_grid.features_importance_
"""