#!/usr/bin/python


import pickle
import numpy as np
import pandas as pd

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


"""
	-Data set information-
	First of all lest find out some data about the loades data set like total number of 
	data point , allocation of target classes (POI vs NON POI), original number of features
	missing values. 
	To accomplish that i will use, as recommended, pandas dataframe

"""

### Load the pickle file
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Copy the loaded data to a new structure to manipulate it
my_dataset = data_dict

### Convert the data set to pandas using float64 as dtype and employee names as index
my_enron_df = pd.DataFrame.from_dict(my_dataset, orient='index', dtype='float64')

### Obtain total number of data points, poi and non poi
print "Data set information"
print "Total number of datapoints : {}".format(len(my_enron_df.index))
print "Allocation of poi / non poi : \n{}".format(my_enron_df['poi'].value_counts())

"""

	-Data cleaning-
	Now its time to clean the dataframe before moving to feature selection.
	Basically I will fill 'NaN' with 0, drop 'TOTAL' index and any other data point 
	which could have all 'NaN' values.
	Adittionally, since i have decided to do a finantial analysis, i will drop from
	the dataframe all mail related features.
	
"""

### Fill 'NaN' with zero first
my_enron_df = my_enron_df.fillna(0)

### Drop 'TOTAL' index and any data point with all 'NaN'
my_enron_df = my_enron_df.drop(['TOTAL'])
my_enron_df = my_enron_df.dropna(how='all')

### Drop email related features 
my_enron_df = my_enron_df.drop(['to_messages','shared_receipt_with_poi','from_messages','from_this_person_to_poi','email_address','from_poi_to_this_person'], axis=1)

### Verify remaining features
print my_enron_df.columns

"""
	-Feature selection-
	After cleaning the dataframe i will start working on feature selection.
	To do that, i will first create a new feature based on the information obtained in the 
	Enron documentary film. Basically my understanding is :

	1) The fraud consisted in "cooking" the accounting books to keep stock prices growing
	2) Executives profits were mostly made out of seeling those overpriced stock, exercise their stock options  
	3) Benefited executives were those who executed those stocks at higher prices -before the crash-

	Based on that i will create a new feature named 'cash_from_stock' which will be the sum of
	exercised_stock_option and restricted_stock. 
	After that, i will use SelectPercentile from scikit learn to find out the 15% of the most
	important features to be used as my_feature_list -and see if cash_from_stock is included as
	per my intuition ;-)-
	http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html 
"""


### Create the new feature based on exercised options related features
my_enron_df['cash_from_stock'] = my_enron_df['exercised_stock_options'] + my_enron_df['restricted_stock']

# print my_enron_df.columns

### Next , i will need to separate 'poi' column as label and the rest of features as features
### To accomplish that i will use dataframe slicing and values attribute to convert the pd.Series to a
### numpy ndarray as requested by scikit learn
labels_series = my_enron_df.loc[:, 'poi']
labels = labels_series.values

### Now the features, we ahould 'index' the slicing with a list of all the features BUT poi
features_columns = []
for col in my_enron_df.columns:
    if col != 'poi':
        features_columns.append(col)

### pandas 'as_matrix' helps convert pandas Series to ndarrays based on selected columns
features = my_enron_df.as_matrix(columns=features_columns)

### Lets check the results. labels should be a 1D array and features a ND array 

print "labels has this shape: {}".format(labels.shape)
print "features has this shape: {}".format(features.shape)

### To do our selection we'll need to fit SelectPercentile in the training data
### Import SelectPercentile and train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

### Training data for feature selection
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=0, test_size=.3)

### OK, 15% of high scored features will be used as my_features_list
select = SelectPercentile(percentile=30)
select.fit(features_train, labels_train)
feats = select.transform(features_train)

### find out which features get selected . mask is a boolean array
mask = select.get_support()

### need to filter the dataframe (only features, no labels) with the mask to get features names
my_enron_df_no_poi = my_enron_df.drop('poi', axis=1)
features_selected = my_enron_df_no_poi.columns[mask]

print "Features selected are: {}".format(features_selected)

"""
	-my_feature_list- SelectPercentile-
	features_list is a list of strings, each of which is a feature name.
	The first feature must be "poi". I will take out exercised_stock_options and total_stock_value
	since they are represented in cash_from_stock (total_stock_value is also represented because is
	actually the sum of exercised + (restricted - restricted deferred) and will include 'salary'
	which is the next weighted feature when moving to 30% percentile
	So, my_feature_list will include the selected features
"""

### 'poi' should be included first to my list

my_features_list = ['poi','salary','cash_from_stock'] 

"""

	-Build the model-
	By this time i have :
	1. A new list of tested features (my_features_list) that includes a new feature (cash_from_stock) 
	2. The labels ('poi' feature in the original dataset)
	3. The features (cleaned features from the original dataset) in the my_enron_df to convert back to 
	pkl by the end of the exploration
	Now is time to convert the dataframe back to the original format and build the models
"""


### Convert my_enron_df back to my_dataset
my_dataset = my_enron_df.to_dict(orient='index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

### I will be testing SVC against RandomForestClassifier using StandardScaler to feature
### scaling , StratifiedShuffleSplit because the amount of samples with poi = 1 is very low
### and train_test_split could end up failing to distribute the class proportional over 
### the train and test sets. Also I will use GridSearchCV for parameters tuning.

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

param_grid = [{'classifier': [SVC()], 'preprocessing': [StandardScaler(), None], 
				'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
				'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
				{'classifier': [RandomForestClassifier(n_estimators=10)],
				 'preprocessing': [None], 'classifier__max_features': [1,2]}]

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#from sklearn.model_selection import train_test_split
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=0)	

### Build the splited sets
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

### Tune parameters for the pipe 
clf = GridSearchCV(pipe, param_grid, cv=cv)
clf.fit(features, labels)
print("Best params:\n{}\n".format(clf.best_params_))
print("Best cross validation score:{:.2f}".format(clf.best_score_))
# print("Test-set score:{:.2f}".format(clf.score(features_test, labels_test)))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features_list)
