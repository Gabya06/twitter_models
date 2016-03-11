# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
import pandas as pd
import os
import sys

from datetime import datetime
import matplotlib.pyplot as pyplot
from sklearn import linear_model, metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_predict
from sklearn import ensemble
from sklearn.utils import shuffle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures



data_path = "/Users/Gabi/Documents/Shareablee/Twitter/data/twitter_data"

pd_read_names = ['index','time', 'impressions', 'retweets', 'page_id', 'twitter_id', 'followers']
pd_read_dtype = [np.str, np.str, np.int, np.int, np.str, np.int, np.int]

pd_kwargs = {
    'header': 0,
    'sep': ',',
    'escapechar': '\\',
    'names': pd_read_names,
    'dtype': dict(zip(pd_read_names, pd_read_dtype)),
    'usecols': ['time', 'impressions', 'retweets', 'page_id', 'followers'],
    'parse_dates' : [0]
}


twitter_data = pd.read_table(data_path, **pd_kwargs)

# print twitter_data.head()
# print twitter_data.dtypes


# followers + retweets = impressions
twitter_data = twitter_data[["page_id", "time" ,"followers","retweets","impressions"]]

twitter_data[['followers','retweets','impressions',]].plot.line(subplots = True, title = '1st look at data')
pyplot.show()

twitter_data[['followers','impressions']].plot.scatter(x = 'followers', y = 'impressions', title = 'followers vs impressions')
pyplot.show()

twitter_data[['retweets','impressions']].plot.scatter(x = 'retweets', y = 'impressions', title = 'retweets vs impressions')
pyplot.show()

# twitter_data[twitter_data.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)].
twitter_data.describe()
print twitter_data.quantile([.1,.25,.5,.75,.95])
# filter out outlier impressions <10% and >90%

retweets_removed = len(twitter_data[(twitter_data.retweets > twitter_data.retweets.quantile(.1)) & (twitter_data.retweets < twitter_data.retweets.quantile(.9))])
impressions_removed = len(twitter_data[(twitter_data.impressions > twitter_data.impressions.quantile(.1)) & (twitter_data.impressions < twitter_data.impressions.quantile(.9))])
print "NUMBER OF OUTLIERS REMOVED:"
retweets_removed + impressions_removed
# remove outliers
twitter_data = twitter_data[(twitter_data.retweets > twitter_data.retweets.quantile(.1)) & (twitter_data.retweets < twitter_data.retweets.quantile(.9))]
twitter_data = twitter_data[(twitter_data.impressions > twitter_data.impressions.quantile(.1)) & (twitter_data.impressions < twitter_data.impressions.quantile(.9))]


# correlations: followers & impressions are more correlated
print "Post cleaning correlations"
print twitter_data[['retweets','followers','impressions']].corr()

# quick look at data
twitter_data[['followers','retweets','impressions',]].plot.line(subplots = True, title = "Post cleaning plots")
pyplot.show()

twitter_data[['followers','impressions']].plot.scatter(x = 'followers', y = 'impressions', title = "Followers vs impressions")
pyplot.show()

twitter_data[['retweets','impressions']].plot.scatter(x = 'retweets', y = 'impressions', title = "Retweets vs impressions")
pyplot.show()

# This is weird
print "This is weird..."
twitter_data.set_index("time")[['followers','impressions']].plot.line(subplots= True)
pyplot.show()

# num of rows
n = twitter_data.shape[0]

# reset new index for data
twitter_index = np.arange(0, twitter_data.shape[0])
twitter_data = twitter_data.set_index(twitter_index)

# add weekday
twitter_data['weekday'] = twitter_data.time.map(lambda x: np.int(x.date().weekday()))
twitter_data.drop('time', axis=1, inplace=True)

# for cross validation
X = twitter_data[['followers','retweets']]
Y = twitter_data.impressions

# # sample for training
n_train = np.round(twitter_data.shape[0]*.90)
ndex_train = np.random.randint(0, n, n_train)

# # training data
x_train	 = twitter_data.ix[ndex_train]
y_train = twitter_data['impressions'][ndex_train]
x_train = x_train[["followers","retweets"]]
#x_train = x_train.drop(['time','page_id','impressions'], axis = 1)

# # test data
x_test = twitter_data.loc[twitter_data.ix != ndex_train]
y_test = twitter_data['impressions'][twitter_data.ix != ndex_train]
x_test = x_test[["followers","retweets"]]
#x_test = x_test.drop(['time','page_id','impressions'], axis = 1)


print "*"*30
print "FEATURE IMPORTANCE"
print "*"*30
X, y = shuffle(twitter_data.drop(['impressions'], axis=1), twitter_data.impressions, random_state=23)

# Use Gradient Boosting Regression method to find relative feature importance
params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
ens_reg = ensemble.GradientBoostingRegressor(**params)
ens_reg.fit(X,y)
feature_importance = ens_reg.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
# plot feature importance
pyplot.figure(figsize=(6,6))
pyplot.barh(pos, feature_importance[sorted_idx], align='center')
pyplot.yticks(pos, X.columns[sorted_idx])
pyplot.xlabel('Relative Importance')
pyplot.title('Variable Importance')
pyplot.show()

# huge
ens_reg.fit(x_train, y_train)
mse = mean_squared_error(y_test, ens_reg.predict(x_test))
print("MSE: %.4f" % mse)


print "*"*30
print "FITTING SIMPLE LINEAR REGRESSION MODEL"
print "*"*30
# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
regrPred = regr.predict(x_test)
mse = mean_squared_error(regr.predict(x_test), y_test)

# # The coefficients
print 'Coefficients: \n', regr.coef_
print "Mean Squared Error: %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2)
# Explained variance score: 1 is perfect prediction (R^2 of prediction)
print'Explained Variance score: %.4f' % regr.score(x_test, y_test)
r2_score(y_test, regrPred)
# # plot actual vs predicted
fig, ax = pyplot.subplots()
ax.scatter(y_test, regrPred)
ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--', lw=4)
ax.set_xlabel('Actual Impressions')
ax.set_ylabel('Predicted Impressions')
pyplot.title("Simple Linear Regression Model")
pyplot.show()


# print "*"*30
# print "CROSS VALIDATION"
# print "*"*30
# # 10-fold cross_validation on entire data
# lr = linear_model.LinearRegression()
# # make prediction
# cvPred = cross_val_predict(estimator = lr, X=X, y=y, cv=10)
# print "R^2 score of cross validation results %0.3f" % r2_score(y, cvPred, multioutput='variance_weighted')
# cvScores = cross_validation.cross_val_score(estimator = lr, X = X, y=Y, cv=10)

# print "Accuracy: %0.3f" % cvScores.mean()


# # # plot actual vs predicted cross validation results
# fig, ax = pyplot.subplots()
# ax.scatter(Y, cvPred)
# ax.plot([Y.min(), Y.max()],[Y.min(), Y.max()], 'r--', lw=4)
# ax.set_xlabel('Actual Impressions')
# ax.set_ylabel('Cross Validation Predicted Impressions')
# pyplot.title("CV results - Linear Regression Model")
# pyplot.show()


print "*"*30
print "RIDGE REGRESSION"
print "*"*30
# RIDGE REGRESSION
ridgeModel = Ridge(alpha = 0.1)
ridgeModel.fit(x_train, y_train)
ridgePred = ridgeModel.predict(x_test)
# # The coefficients
print 'Coefficients: \n', ridgeModel.coef_
print "Residual sum of squares: %.2f" % np.mean((ridgePred - y_test) ** 2)
# Explained variance score: 1 is perfect prediction
print 'Variance score: %.2f' % ridgeModel.score(x_test, y_test)
fig, ax = pyplot.subplots()
ax.scatter(y_test, ridgePred)
ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--',lw=4)
ax.set_xlabel('Actual Impressions')
ax.set_ylabel('Predicted Impressions')
pyplot.title("Ridge Regression Model")
pyplot.show()

#pyplot.semilogx(alphas, scores)
# plot error lines showing +/- std. errors of the scores
# pyplot.semilogx(alphas, np.array(scores) + np.array(scores_std) / np.sqrt(len(X)), 'b--')
# pyplot.semilogx(alphas, np.array(scores) - np.array(scores_std) / np.sqrt(len(X)), 'b--')

print "*"*30
print "CROSS VALIDATION/K-FOLD FOR LASSO REGRESSION"
print "*"*30

lassoModel = linear_model.Lasso()
alphas = np.logspace(-4, -.5, 30)
scores = list()
scores_std = list()
for alpha in alphas:
    lassoModel.alpha = alpha
    this_scores = cross_validation.cross_val_score(estimator = lassoModel, X=X, y=Y, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

pyplot.figure(figsize=(6, 6))
pyplot.semilogx(alphas, scores)
pyplot.ylabel('CV score')
pyplot.xlabel('alpha')
pyplot.axhline(np.max(scores), linestyle='--', color='red')
pyplot.show()

lasso_cv = linear_model.LassoCV(alphas=alphas, cv = 5, selection = 'random')
folds = 30
k_fold = cross_validation.KFold(n=len(X), n_folds=folds, shuffle=True)
cv_scores = list()
cv_alphas = list()
for k, (train, test) in enumerate(k_fold):
	lasso_cv.fit(X.ix[train], Y.ix[train])
	lasso_cv.alpha_ = alphas[k]
	cv_alphas.append(lasso_cv.alpha_)
	cv_scores.append(lasso_cv.score(X.ix[test], Y.ix[test]))
	print("[fold {0}] alpha: {1:.9f}, score: {2:.5f}". format(k, lasso_cv.alpha_, lasso_cv.score(X.ix[test], Y.ix[test])))
#    lasso_cv.fit(x_train, y_train)
    # cv_scores[k] = lasso_cv.score(x_test, y_test)
    # cv_alphas[k] = lasso_cv.alpha_

# put results in dataframe
lasso_cv_df = pd.DataFrame({'fold': range(folds),'alpha': cv_alphas, 'score': cv_scores})
# plot lasso cv results
pyplot.figure()
pyplot.semilogx(cv_alphas, cv_scores, 'r--')
pyplot.ylabel('CV score')
pyplot.xlabel('alpha')
pyplot.title('Cross Validation Lasso Regression')
pyplot.show()


# best alpha
print "Best alpha's\n", lasso_cv_df.sort_values('score', ascending=False).head(10)
best_alpha = lasso_cv_df.sort_values('score', ascending=False).alpha.iloc[0]

lassoModel = linear_model.Lasso(alpha = best_alpha)
lassoModel.fit(x_train, y_train)
lassoPred = lassoModel.predict(x_test)
# # The coefficients
print('Coefficients: \n', lassoModel.coef_)
print("Residual sum of squares: %.2f" % np.mean((lassoPred - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lassoModel.score(x_test, y_test))
r2_score(lassoPred, y_test)

fig, ax = pyplot.subplots()
ax.scatter(y_test, lassoPred)
ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--',lw=4)
ax.set_xlabel('Actual Impressions')
ax.set_ylabel('Predicted Impressions')
pyplot.title("Lasso Regression Model")
pyplot.show()




# ax = pyplot.subplot(1,4,1)
# ax.scatter(y_test, regrPred)
# ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--', lw=4)
# ax.set_xlabel('Actual Impressions')
# ax.set_ylabel('Predicted Impressions')
# pyplot.title("Simple Linear Regression Model", color = 'blue')

# ax = pyplot.subplot(1,4,2)
# ax.scatter(Y, cvPred)
# ax.plot([Y.min(), Y.max()],[Y.min(), Y.max()], 'r--', lw=4)
# ax.set_xlabel('Actual Impressions')
# ax.set_ylabel('Cross Validation Predicted Impressions')
# pyplot.title("CV results - Linear Regression Model")

pyplot.figure(figsize=(12,10))
ax =pyplot.subplot(1,2,1)
ax.scatter(y_test, ridgePred)
ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--',lw=4)
ax.set_xlabel('Actual Impressions')
ax.set_ylabel('Predicted Impressions')
pyplot.title("Ridge Regression Model" + "\n" + str(np.round(ridgeModel.coef_[0],5)) + " " + str(np.round(ridgeModel.coef_[1])))

ax =pyplot.subplot(1,2,2)
ax.scatter(y_test, lassoPred)
ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'r--',lw=4)
ax.set_xlabel('Actual Impressions')
ax.set_ylabel('Predicted Impressions')
pyplot.title("Lasso Regression Model" + "\n" + str(np.round(lassoModel.coef_[0],5)) + " " + str(np.round(lassoModel.coef_[1])))
pyplot.show()





# from sklearn.linear_model import ElasticNet
# enet = ElasticNet(alpha=best_alpha, l1_ratio=0.7)

# y_pred_enet = enet.fit(x_train, y_train).predict(x_test)
# r2_score_enet = r2_score(y_test, y_pred_enet)
# print(enet)
# print("r^2 on test data : %f" % r2_score_enet)
