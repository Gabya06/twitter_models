#! /usr/bin/python
# -*- coding: utf-8 -*-

#http://connor-johnson.com/2014/02/18/linear-regression-with-python/

"""
script for buidling linear model

@author: Gabi
"""


from load import *


class Model:
	def __init__(self):
		self.tw_data = load_tw_data()
		self.user_data = load_user_data()

		# self.result = pd.DataFrame(columns = ['base_user','cross_user','base_total_overlap','perc_overlap'])
		self.model_results = pd.DataFrame(columns = ['page_id','followers','retweets','impressions','predicted', \
													'perc_diff', 'err_cat'])
		self.rmse = None
		self.r_score = None
		self.n = None
		self.model = None



	def clean(self, quantile):
		self.tw_data = self.tw_data[(self.tw_data.retweets > self.tw_data.retweets.quantile(quantile)) & (self.tw_data.retweets < self.tw_data.retweets.quantile(1.0-quantile))]
		self.tw_data = self.tw_data[(self.tw_data.impressions > self.tw_data.impressions.quantile(quantile)) & (self.tw_data.impressions < self.tw_data.impressions.quantile(1.0-quantile))]
		# reset new index for data
		self.n = self.tw_data.shape[0]
		# twitter_index = np.arange(0, self.n)
		self.tw_data = self.tw_data.set_index(np.arange(0, self.n))

		# add weekday
		try:
			self.tw_data['weekday'] = self.tw_data.time.map(lambda x: np.int(x.date().weekday()))
			self.tw_data.drop('time', axis=1, inplace=True)

		except:
			pass
		# if 'time' in self.tw_data.columns: self.tw_data.drop('time', axis=1, inplace=True)


	def _get_train_test(self, perc_train = 0.9):
		# # sample for training
		n_train = np.round(self.n * perc_train)
		ndex_train = np.random.randint(0, self.n, n_train)

		# # training data
		x_train	 = self.tw_data.ix[ndex_train]
		y_train = self.tw_data['impressions'][ndex_train]
		x_train = x_train[["followers","retweets"]]

		# # test data
		x_test = self.tw_data.drop(ndex_train, axis = 0)
		y_test = x_test[['impressions']]

		self.model_results[['page_id','followers','retweets','impressions']] = x_test[['page_id','followers','retweets','impressions']].copy()
		x_test = x_test[["followers","retweets"]]


		return [x_train, y_train, x_test, y_test]



	def cross_validate(self, alphas, folds):
		model_cv = linear_model.RidgeCV(alphas=alphas, cv = 5)
		k_fold = cross_validation.KFold(n=self.n, n_folds=folds, shuffle=True)
		cv_scores = list()
		cv_alphas = list()
		for k, (train, test) in enumerate(k_fold):
			# X = self.tw_data.drop(['tw_name','impressions'], axis=1)
			X = self.tw_data.drop(['impressions'], axis=1)
			Y = self.tw_data.impressions
			model_cv.fit(X.ix[train], Y.ix[train])
			model_cv.alpha_ = alphas[k]
			cv_alphas.append(model_cv.alpha_)
			cv_scores.append(model_cv.score(X.ix[test], Y.ix[test]))
			print("[fold {0}] alpha: {1:.9f}, score: {2:.5f}". format(k, model_cv.alpha_, model_cv.score(X.ix[test], Y.ix[test])))
		model_cv_df = pd.DataFrame({'fold': range(folds),'alpha': cv_alphas, 'score': cv_scores})
		print "Best alpha's\n", model_cv_df.sort_values('score', ascending=False).head(10)
		#print "Best alpha's\n", model_cv_df.sort_values('score', ascending=False).head(10)
		best_alpha = model_cv_df.sort_values('score', ascending=False).alpha.iloc[0]
		return best_alpha


	def train(self, model, perc_train, alpha):
		x_train, y_train, x_test, y_test = self._get_train_test(perc_train)
		self.model = model(alpha)
		self.model.fit(x_train, y_train)
		self.r_score = self.model.score(x_test, y_test)
		self.model_results.predicted = self.model.predict(x_test)


	def get_coefs(self):
		return self.model.coef_

	def get_results(self):
		tw_samples = self.user_data[['user_id','tw_name']].drop_duplicates()
		tw_samples.rename(columns = {'user_id':'page_id'}, inplace=True)

		diff = np.subtract(self.model_results.impressions, self.model_results.predicted)
		self.model_results.perc_diff = abs(diff/self.model_results.impressions)

		bins = np.arange(0.0, 1.0, .3)
		large_bins = np.arange(1.0, self.model_results.perc_diff.max()+5, 4)
		all_bins = np.concatenate([bins, large_bins])

		self.model_results['err_cat'] =  pd.cut(self.model_results.perc_diff, bins = all_bins, right = True)
		self.model_results = self.model_results.merge(tw_samples, on = 'page_id')

 		results = self.model_results.groupby(['page_id','tw_name','err_cat'])[['err_cat']].agg('count')
 		results.rename(columns = {'err_cat':'frequency'}, inplace = True)
 		results = results.reset_index()
 		results.sort_values(['frequency','tw_name'], ascending=False, inplace=True)
 		results['perc_total_err'] = results.groupby('err_cat')['frequency'].apply(lambda x: x/(x.sum()))
 		return results


def plot_errors(results):

	import matplotlib as mpl
	label_size = 9
	mpl.rcParams['xtick.labelsize'] = label_size

	results.tw_name.replace('theeconomist',"economist", inplace=True)
	results.tw_name.replace('fortunemagazine',"fortune_mag", inplace=True)
	results.tw_name.replace('cartoonnetwork','cartoon_net', inplace=True)
	results.tw_name.replace('hallmarkchannel','hallmark_ch', inplace=True)
	results.tw_name.replace('bet','BET', inplace=True)
	results.tw_name.replace('TIME','Time', inplace=True)
	results.tw_name.replace('DIRECTV','DirecTV', inplace=True)

	# cut errors into bins
	bins = results.err_cat.drop_duplicates()
	bins = bins.reset_index().drop('index',axis=1)
	# plot
	n_bins = results.err_cat.drop_duplicates().shape[0]
	n_cols = int(np.round(n_bins/2.0))
	fig, axes = pyplot.subplots(nrows = 2, ncols = n_cols, figsize = (16,8))
	for i, var in enumerate(bins.err_cat):
		if i <=n_cols-1:
			ax = axes[0][i]
		elif i>n_cols-1:
			ax = axes[1][i-n_cols-1]

		tmp = results.loc[(results.err_cat == var)] # & (results.frequency>100)]
		if (any(tmp[tmp.frequency >100].frequency) & tmp[tmp.frequency >100].shape[0]>=3):
			tmp = results.loc[(results.err_cat == var) & (results.frequency>100)]
		else:
			tmp = results.loc[(results.err_cat == var)] # & (results.frequency>100)]


		tmp[['tw_name', 'frequency']].plot(kind = 'bar', x ='tw_name', y = 'frequency', ax = ax, title = var + " error", legend = '')
		ax.set_xlabel('')

	axes[0][0].set_ylabel('error frequency', rotation = 90, labelpad = 20, fontsize = 8)
	axes[1][0].set_ylabel('error frequency', rotation = 90, labelpad = 20, fontsize = 8)
		#errors.plot(kind = 'bar', x= 'err', y = 'count')
	fig.subplots_adjust(hspace=0.25, wspace=0.25)
	pyplot.tight_layout()
	pyplot.show()

# new_cols = pd.get_dummies(x_train.tw_name)
# x_train = pd.concat([x_train, new_cols], axis = 1)
# x_train = x_train.drop('tw_name', axis=1)
# bins = np.arange(0, max(ridgeResults.perc_diff)+.3, .3)
# ridgeResults['err_cat'] =  pd.cut(ridgeResults.perc_diff, bins = bins, right = True)
# large_errors = pd.DataFrame(ridgeResults[ridgeResults.perc_diff >.75].page_id.unique(), columns = ['page_id'])
# large_errors = large_errors.merge(tw_samples, on='page_id')

# large_errors = user_data[user_data.user_id.isin(large_errors_id)][['tw_name']].drop_duplicates()


# errors = ridgeResults.perc_diff.unique().tolist()
# errors = [e for e in errors if e >1.0]

# ridgeResults.err_cat =  pd.cut(ridgeResults.perc_diff, bins = bins, right = True)
# bins2 = np.linspace(min(errors), max(errors)+1, num = 10)


# print "*" * 15
# print "MODEL 1 - Using all 27 properties for modeling impressions"
# linModel1 = Model()
# linModel1.clean(quantile = .1)
# print "TRAINING ...."
# linModel1.train(model = Ridge, perc_train = .9, alpha = .1)
# print "\n"
# print "GETTING RESULTS ...."
# score = linModel1.r_score
# print "ADJUSTED R2 = ", score
# print "\n"
# results = linModel1.get_results()
# plot_errors(results)
# print "*" * 15


# err_threshold = 0.6
# model_err_stats = linModel1.model_results.groupby('tw_name')['perc_diff'].agg(['count','sum','mean']).sort_values('mean',ascending = False, axis =0).reset_index()
# model_err_stats.rename(columns = {'count':'frequency', 'sum':'err_sum' ,'mean':'err_mean'}, inplace = True)
# tw_names_drop = model_err_stats[model_err_stats.err_mean > err_threshold].tw_name.drop_duplicates()



# cross validation
# alphas = np.logspace(-4, -.5, 30)
# print "*" * 15
# print "MODEL 2"
# print "REMOVING TWITTER USERS WITH AVG ERRORS > 60%"
# print tw_names_drop

# linModel2 = Model()
# # remove names with largest average of error
# print linModel2.tw_data.head()
# linModel2.tw_data = linModel2.tw_data[~linModel2.tw_data.tw_name.isin(tw_names_drop)]
# linModel2.clean(quantile = .1)
# print "CROSS VALIDATION ...."
# best_alpha = linModel2.cross_validate(alphas=alphas, folds=5)
# print "BEST ALPHA: ", best_alpha
# print "TRAINING ...."
# linModel2.train(model = Ridge, perc_train = .9, alpha = best_alpha)
# score2 = linModel2.r_score
# print "ADJUSTED R2 = ", score2
# results2 = linModel2.get_results()
# print "*" * 15
# tw_names_drop = linModel.model_results[linModel.model_results.perc_diff>5].tw_name.drop_duplicates()
# linModel2.tw_data = linModel2.tw_data[~linModel3.tw_data.tw_name.isin(tw_names_drop)]
# linModel2.clean(quantile = .1)
# linModel2.train(model = Ridge, perc_train = .9, alpha = .1)

# print "*" * 30
# print "MODEL 1 - Using all 27 properties for modeling impressions"
# print ".... CLEANING DATA .... REMOVING OUTLIERS ...."
# linModel1 = Model()
# linModel1.clean(quantile = .1)
# print ".... TRAINING MODEL 1 ...."
# linModel1.train(model = Ridge, perc_train = .9, alpha = .1)
# print "\n"
# print ".... GETTING RESULTS FOR MODEL 1...."
# score = linModel1.r_score
# print "ADJUSTED R2 = ", score
# print "\n"
# results = linModel1.get_results()

# err_threshold = 0.6
# model_err_stats = linModel1.model_results.groupby(['page_id','tw_name'])['perc_diff'].agg(['count','sum','mean']).sort_values('mean',ascending = False, axis =0).reset_index()
# model_err_stats.rename(columns = {'count':'frequency', 'sum':'err_sum' ,'mean':'err_mean'}, inplace = True)
# tw_names_drop = model_err_stats[model_err_stats.err_mean > err_threshold][['page_id','tw_name']].drop_duplicates()

# print "MODEL 1 RESULTS:"
# print model_err_stats
# print
# print "BIGGEST ERRORS:"
# print tw_names_drop

# print ".... PLOTTING ERRORS ...."

# plot_errors(results)


# cross validation for model 2 (exludes users with largest errors)
# alphas = np.logspace(-4, -.5, 20)
# print "*" * 30
# print "MODEL 2"
# print "REMOVING TWITTER USERS WITH AVG ERRORS > 60%"


# linModel2 = Model()
# # remove names with largest average of error
# linModel2.tw_data = linModel2.tw_data[~linModel2.tw_data.page_id.isin(tw_names_drop.page_id)]
# print ".... CLEANING DATA .... REMOVING OUTLIERS ...."
# linModel2.clean(quantile = .1)
# print "\n"
# print "5 FOLD CROSS VALIDATION ...."
# best_alpha = linModel2.cross_validate(alphas=alphas, folds=5)
# print
# print "FOUND BEST ALPHA USED IN MODEL 2: ", best_alpha
# print ".... TRAINING MODEL 2...."
# linModel2.train(model = Ridge, perc_train = .9, alpha = best_alpha)
# score2 = linModel2.r_score
# print "ADJUSTED R2 = ", score2
# results2 = linModel2.get_results()

# model_err_stats2 = linModel2.model_results.groupby(['page_id','tw_name'])['perc_diff'].agg(['count','sum','mean']).sort_values('mean',ascending = False, axis =0).reset_index()
# model_err_stats2.rename(columns = {'count':'frequency', 'sum':'err_sum' ,'mean':'err_mean'}, inplace = True)
# tw_names_drop2 = model_err_stats2[model_err_stats2.err_mean > err_threshold][['page_id','tw_name']].drop_duplicates()

# print "MODEL 2 RESULTS:"
# print model_err_stats2
# print
# print "BIGGEST ERRORS:"
# print tw_names_drop2

# print ".... PLOTTING ERRORS FOR MODEL 2...."
# plot_errors(results2)



# print "*" * 30
# print "BUILDING MODEL FOR TWITTER USERS GUILTY OF LARGEST ERRORS"

# linModel3 = Model()
# linModel3.tw_data = linModel3.tw_data[linModel3.tw_data.page_id.isin(tw_names_drop.page_id)]
# linModel3.clean(quantile = .1)
# best_alpha_3 = linModel3.cross_validate(alphas=alphas, folds=10)
# linModel3.train(model = Ridge, perc_train = .9, alpha = best_alpha_3)
# score3 = linModel3.r_score
# results3 = linModel3.get_results()
# print results3
# model_err_stats3 = linModel3.model_results.groupby(['page_id','tw_name'])['perc_diff'].agg(['count','sum','mean']).sort_values('mean',ascending = False, axis =0).reset_index()
# model_err_stats3.rename(columns = {'count':'frequency', 'sum':'err_sum' ,'mean':'err_mean'}, inplace = True)

# print "MODEL 3 RESULTS:"
# print model_err_stats3

# print ".... PLOTTING ERRORS FOR MODEL 3...."
# plot_errors(results3)




