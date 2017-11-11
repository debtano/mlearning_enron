README.md

Context of the Data  
--------------------

My first objective is to get some info on the Enron case and understand what have happened to be able 
to look at the available data with the right context. To do that I watched, as recommended, the TV documentary : 
"Enron : the smartest guys in the rooms" . Based on that, I coul conclude the following statements about
my comprehension to help me start wrinting somo hipothesis and some answers :

1) The fraud consisted in "cooking" the accounting books to keep stock prices growing
2) Executives profits were mostly made out of seeling those overpriced stock, exercise their stock options  
3) Benefited executives were those who executed those stocks at higher prices -before the crash-
4) We have available 4 features that are directly related with item #3 which are :

total_stock_value
exercised_stock_options
restricted_stock_deferred
restricted_stock

where total_stock_value = exercised_stock_option + (restricted_stock - restricted_stock_deferred)

So, based on that hypothesis, my first task will be to explore the dataset to extract basic informarion and the 
second one to explore those features.

Machine learning techniques are very usefull in this case basically because we have a dataset that includes features and labels (poi or non poi); our objective is to predict, based on some features values, if the sample (enron employee in this case) can be categorized as poi or non poi. For that reasson, since we have data already categorized, our problem is a binary (poi / nor poi), classification (discrete, no regression), supervised (train set with features and labels) machine learning problem.

Data exploration and outliers
-----------------------------

Based on the hypothesis mentioned earlier i will try to check it using SelectPercentile from sklearn to validate what are
the most important features to classify POI. 
Before that i have cleaned the data -please check doc in the script- dealing with 'NaN', summarized fields, data points with
all 'NaN' and features related with mail which i wont use.

For my new feature list i will include the ones selected by SelectPercentile including a new feature i created which is the sum
of cash from exercised stock and restricted stock (cash_from_stock)

For dataset information please check the initial part of the script. Based on the recommendation i created a pandas dataframe
based on the initial structure to find out number of data points, number of points with poi/non poi class .

Algorithm and Parameters
------------------------

Since this is a supervised binary classification problem i checked several algorithms used in tha field : gaussianNB, k-neighbords, RandomTreesClasiffier, SVC and LogisticRegression (i provided the jupyter notebook as backup data where i was "playing around" with the aforementioned algoriths).

Based on the above mentioned tests, i choose to compare performance between an algorithm that requires preprocessing (scaling) like SVC and an algorithm that did not -and had good results when checked- : RandomForestClassifier.

To compare performance of both choosen algorithms I used GridSearchCV that includes cross-validation. Based on the recommendations , and since the poi class has a low presence on the whole dataset i will use StratifiedShuffleSplit to
split the dataset instead of train_test_split.

For the tuning parameters i was guided by a specialized publication (Andreas Muller - see REFERENCES). So i used param_grid to let the optimizer look for the best C and gamma parameters for SVC. 

Tunning of parameters is very import because every machine learning problem has a point of equilibrium for the model where additional attention and fitting will overfit the model (which means good precission for existing samples but not so good generalization for new ones) or too little information which produce a model underfitting . Another tradeoff , this is more related with the RandomForestClassifier case, is when you have an algorithm than can run , parallel or serialy, an amount of iterations processing the information and adjusting itself; in this case the tradeoff is between amount of resources and time to train -more iterations- and precission.

I learned that GridSearch + CV is a good strategy to combine algorithm optimization and cross validation. Cross validation basically try to avoid the mistake of manually separate train and test data and then find out that the test data did not clearly and fair represents the train data so your model wont be trained as required.

I also include a StandardScaler in the pipeline for feature scaling.

Metrics
-------

Given the objective for the work which was to obtain at least 0.3 precision and recall my intention was to put focus on recall over precision -keeping both under 0.3- . 

The reasson for this was that my understanding is that recall will help you identify all positives minimizing false negatives; in this case we are trying to identify persons of interest which are probably not parto of the fraud but further investigation is required so i want to identify them first based on finantial data and then move on understanding if it was a false positive. Precision helps more than recall when your objective is to minimize false positives -like monitoring alarms- in that case "further investigation" is a problem because you should take action to the alarm inmediatly.


