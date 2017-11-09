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

Based on the hypothesis mentioned earlier i will relate 'salary' with a new feature which will be the sum of 'exercised_stock_options' and 'restricted_stock' and i will call it 'cash_from_stock'. This new feature will be created through a provided function 'combineStock' inside the 'clean_and_combine' package.
I first checked adding together salary + bonus + incentive  but (it could be found in the jupiter notebook material) as i added more features with so many 'NaN' -salary at least is a very populated feature- the precission got worst.
Outliers will be managed by including the StandardScaler in the pipe.
There are lot of missing values for the features to be used; they will be managed using a 'cleanNaN' function inside the provided clean_and_combine package to replace 'NaN' by 0.

The dataset contains 146 samples (includind a 'TOTAL' sample that is deleted in the clean stage) and 21 features

Algorithm and Parameters
------------------------

Since this is a supervised binary classification problem i checked several algorithms used in tha field : gaussianNB, k-neighbords, RandomTreesClasiffier, SVC and LogisticRegression (i provided the jupyter notebook as backup data where i was "playing around" with the aforementioned algoriths).
Based on the above mentioned tests, i choose to compare performance between an algorithm that requires preprocessing (scaling) like SVC and an algorithm that did not -and had good results when checked- : RandomForestClassifier.
To compare performance of both choosen algorithms I used GridSearchCV that includes cross-validation.
For the tuning parameters i was guided by a specialized publication (Andreas Muller - see REFERENCES). So i used param_grid to let the optimizer look for the best C and gamma parameters for SVC. 
Tunning of parameters is very import because every machine learning problem has a point of equilibrium for the model where additional attention and fitting will overfit the model (which means good precission for existing samples but not so good generalization for new ones) or too little information which produce a model underfitting . Another tradeoff , this is more related with the RandomForestClassifier case, is when you have an algorithm than can run , parallel or serialy, an amount of iterations processing the information and adjusting itself; in this case the tradeoff is between amount of resources and time to train -more iterations- and precission.

I learned that GridSearch + CV is a good strategy to combine algorithm optimization and cross validation. Cross validation basically try to avoid the mistake of manually separate train and test data and then find out that the test data did not clearly and fair represents the train data so your model wont be trained as required.

Metrics
-------

Given the objective for the work which was to obtain at least 0.3 precision and recall my intention was to put focus on recall over precision -keeping both under 0.3- . The reasson for this was that my understanding is that recall will help you identify all positives minimizing false negatives; in this case we are trying to identify persons of interest which are probably not parto of the fraud but further investigation is required so i want to identify them first based on finantial data and then move on understanding if it was a false positive. Precision helps more than recall when your objective is to minimize false positives -like monitoring alarms- in that case "further investigation" is a problem because you should take action to the alarm inmediatly.


