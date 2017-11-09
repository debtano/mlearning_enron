#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from clean_and_combine import cleanNan, combineStock

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### New features list
my_features_list = ['poi', 'salary', 'cash_from_stock']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Clean 'TOTAL' sample, create new features and clean NaN
del my_dataset['TOTAL']
cleanNan(my_dataset)
combineStock(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

param_grid = [{'classifier': [SVC()], 'preprocessing': [StandardScaler(), None], 
				'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
				'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
				{'classifier': [RandomForestClassifier(n_estimators=100)],
				 'preprocessing': [None], 'classifier__max_features': [1,2]}]

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=0)	

# Example starting point. Try investigating other evaluation techniques!
## from sklearn.cross_validation import train_test_split
## features_train, features_test, labels_train, labels_test = \
##    train_test_split(features, labels, test_size=0.3, random_state=42)

kfold = KFold(n_splits=5)

clf = GridSearchCV(pipe, param_grid, cv=kfold)
clf.fit(features_train, labels_train)
print("Best params:\n{}\n".format(clf.best_params_))
print("Best cross validation score:{:.2f}".format(clf.best_score_))
print("Test-set score:{:.2f}".format(clf.score(features_test, labels_test)))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features_list)