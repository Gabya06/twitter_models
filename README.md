# Twitter model 

## Project Overview

This repository is used to model twitter impressions using followers and retweets, based on linear regression models (Rigde Regression in particular)

The python code in the jupyter notebook is used to predict twitter impressions based on retweets and followers. The model is used to initialize, clean data, train, predict and preform cross validation. Three slightly different models are built:

## Model 1:
The 1st model is built using all 27 properties and by remove top and bottom 10% of retweets and followers. Ridge regression is used with alpha set to 0.1, and R-squared is 0.764 using 90% training and 10% test data. Using an average error threshold of 60% filters out properties with larger average errors such as CNN, NFL, MTV, BET and InStyle. Performing 5-fold cross validation to find the best alpha for ridge regression yields about the same score with slightly lower average errors.

```python
print "*" * 30
print "MODEL 1 - Using all 27 properties for modeling impressions"
print ".... CLEANING DATA .... REMOVING OUTLIERS ...."
linModel1 = Model()
linModel1.clean(quantile = .1)

print ".... TRAINING MODEL 1 ...."
linModel1.train(model = Ridge, perc_train = .9, alpha = .1)

print ".... GETTING RESULTS FOR MODEL 1...."
score = linModel1.r_score
print "ADJUSTED R2 = ", score

results = linModel1.get_results()
print "Model Coefficients:"
coefs = linModel1.get_coefs()
print coefs
```

ADJUSTED R2 =  0.765818282025

5-fold cross validation scores:

- [fold 0] alpha: 0.000100000, score: 0.76615
- [fold 1] alpha: 0.000152831, score: 0.76248
- [fold 2] alpha: 0.000233572, score: 0.76689
- [fold 3] alpha: 0.000356970, score: 0.75942
- [fold 4] alpha: 0.000545559, score: 0.76597

Best alpha's

|alpha     |fold     |score |
|----------|:-------:|:----:|
|0.000234  |   2     |0.7668|
|0.000100  |   0     |0.7661|
|0.000546  |   4     |0.7659|
|0.000153  |   1     |0.7624|
|0.000357  |   3     |0.7594|

Properties with the largest errors in model 1:

CNN, NFL, MTV, 106andpark, BET, InStyle, Brueggers, Dasaniwater, Essencemag, ABC11 wtvd


## Model 2:
The 2nd model removes properties that had the highest average error (over 60%) and also performs 5-fold cross-validation and searches for best alpha for Ridge regression. The average score on the folds is higher than the 1st model (around 0.82). The largest errors are just at 50% with smaller properties, which leads me to think that maybe we need to build a model for properties such as CNN and MTV and another model for smaller properties.

```python
alphas = np.logspace(-4, -.5, 20)
print "*" * 30
print "MODEL 2"
print "REMOVING TWITTER USERS WITH AVG ERRORS > 60%"


linModel2 = Model()
# remove names with largest average of error
linModel2.tw_data = linModel2.tw_data[~linModel2.tw_data.page_id.isin(tw_names_drop.page_id)]
print ".... CLEANING DATA .... REMOVING OUTLIERS ...."
linModel2.clean(quantile = .1)
print "\n"
print "5 FOLD CROSS VALIDATION ...."
best_alpha = linModel2.cross_validate(alphas=alphas, folds=5)
print
print "FOUND BEST ALPHA USED IN MODEL 2: ", best_alpha
print ".... TRAINING MODEL 2...."
linModel2.train(model = Ridge, perc_train = .9, alpha = best_alpha)
score2 = linModel2.r_score
print "ADJUSTED R2 = ", score2
results2 = linModel2.get_results()
print "Model Coefficients:"
coefs2 = linModel2.get_coefs()
print coefs2
```

ADJUSTED R2 =  0.816688528879

5-FOLD CROSS VALIDATION SCORES:
- [fold 0] alpha: 0.000100000, score: 0.81777 <br>
- [fold 1] alpha: 0.000152831, score: 0.82348 <br>
- [fold 2] alpha: 0.000233572, score: 0.82171
- [fold 3] alpha: 0.000356970, score: 0.81639
- [fold 4] alpha: 0.000545559, score: 0.81868

Best alpha's

|alpha     |fold     |score |
|----------|:-------:|:----:|
|0.000153  |  1      |0.8234|
|0.000234  |  2      |0.8217|
|0.000546  |  4      |0.8186|
|0.000100  |  0      |0.8177|
|0.000357  |  3      |0.8163|

## Model 3:
The 3rd model is built using only those properties with the largest errors (over 60%) and the average R-squared error is closer to 0.89, so this model outperforms the other two. While there are still some properties with large errors, some like CNN do a better job using this model than the 1st model which includes all properties.

```python
print "*" * 30
print "MODEL 3"

print "*" * 30
print "BUILDING MODEL FOR TWITTER USERS GUILTY OF LARGEST ERRORS"

linModel3 = Model()
linModel3.tw_data = linModel3.tw_data[linModel3.tw_data.page_id.isin(tw_names_drop.page_id)]
print ".... CLEANING DATA .... REMOVING OUTLIERS ...."
linModel3.clean(quantile = .1)
print "\n"
print "10 FOLD CROSS VALIDATION ...."
best_alpha_3 = linModel3.cross_validate(alphas=alphas, folds=10)
print
print "FOUND BEST ALPHA USED IN MODEL 3: ", best_alpha_3
print ".... TRAINING MODEL 3...."
linModel3.train(model = Ridge, perc_train = .9, alpha = best_alpha_3)
score3 = linModel3.r_score
print "ADJUSTED R2 = ", score3
print "Model Coefficients:"
coefs3 = linModel3.get_coefs()
print coefs3
```

ADJUSTED R2 =  0.888740204144

10 FOLD CROSS VALIDATION SCORES:

- [fold 0] alpha: 0.000100000, score: 0.89105
- [fold 1] alpha: 0.000152831, score: 0.89236
- [fold 2] alpha: 0.000233572, score: 0.88658
- [fold 3] alpha: 0.000356970, score: 0.89746
- [fold 4] alpha: 0.000545559, score: 0.88830
- [fold 5] alpha: 0.000833782, score: 0.89534
- [fold 6] alpha: 0.001274275, score: 0.88810
- [fold 7] alpha: 0.001947483, score: 0.89164
- [fold 8] alpha: 0.002976351, score: 0.89149
- [fold 9] alpha: 0.004548778, score: 0.89680

Best alpha's

|alpha     |fold     |score |
|----------|:-------:|:----:|
|0.000357  | 3       |0.8974|
|0.004549  | 9       |0.8967|
|0.000834  |  5      |0.8953|
|0.000153  |  1      |0.8923|
|0.001947  |  7      |0.8916|
|0.002976  |  8      |0.8914|
|0.000100  |  0      |0.8910|
|0.000546  |  4      |0.8883|
|0.001274  |  6      |0.8881|
|0.000234  |  2      |0.8865|


### A bit of documentation on Model class:
The Model class initializes a model with the following attributes:
Twitter and user data 
Model results: a dataframe to store page_id, followers, retweets, impressions, predicted and percent differences
rmse score
n data points
model (linear regression)
The clean function:
Removes retweets and impressions above and below the input quantile (this is .10 by default)
Add weekday as a feature and drop time column
The train function gets randomized training and test data based on percent to train (default is 90%), fits the model and sets the score and predicted values.
The cross validate function takes as input alphas and the number of k-folds to run (default is 5) and performs cross-validation to find the best alpha. It returns the best alpha found based on k-fold cross validation.
