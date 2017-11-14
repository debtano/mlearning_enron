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
my_dataset = my_dataset.to_dict(orient='index')

### First we will train a model without new features . We'll use the "original_features" list
### which cointains what features remain after cleaning mail features. Lets print the list
print "\n\n This is the list of features for the first train and validation: {}".format(initial_features)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, initial_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Initially, i wont tune parameters for the RandomForestClassifier, i want to understand 
### features importance first
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier


