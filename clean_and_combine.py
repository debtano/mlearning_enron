#!/usr/bin/python

### Features to be combined and cleaned
features = ['salary', 'exercised_stock_options','restricted_stock']

def cleanNan(my_dataset):
	keys = sorted(my_dataset.keys())
	for key in keys:
		for feature in features:
			if my_dataset[key][feature]=='NaN':
				value = 0
				my_dataset[key][feature] = float( value )

def combineStock(my_dataset):
	keys = sorted(my_dataset.keys())
	for key in keys:
		exercised = my_dataset[key]['exercised_stock_options']
		restricted = my_dataset[key]['restricted_stock']
		my_dataset[key]['cash_from_stock'] = exercised + restricted

