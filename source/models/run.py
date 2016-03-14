'''
script for loading data and importing libraries
@author: Gabi
'''

# -*- coding: utf-8 -*-
#!/usr/bin/python


from twitter_models import *

print "*" * 30
print "MODEL 1 - Using all 27 properties for modeling impressions"
print ".... CLEANING DATA .... REMOVING OUTLIERS ...."
linModel1 = Model()
linModel1.clean(quantile = .1)
print ".... TRAINING MODEL 1 ...."
linModel1.train(model = Ridge, perc_train = .9, alpha = .1)
print "\n"
print ".... GETTING RESULTS FOR MODEL 1...."
score = linModel1.r_score
print "ADJUSTED R2 = ", score
print "\n"
results = linModel1.get_results()
print "Model Coefficients:"
coefs = linModel1.get_coefs()
print coefs

err_threshold = 0.6
model_err_stats = linModel1.model_results.groupby(['page_id','tw_name'])['perc_diff'].agg(['count','sum','mean']).sort_values('mean',ascending = False, axis =0).reset_index()
model_err_stats.rename(columns = {'count':'frequency', 'sum':'err_sum' ,'mean':'err_mean'}, inplace = True)
tw_names_drop = model_err_stats[model_err_stats.err_mean > err_threshold][['page_id','tw_name']].drop_duplicates()

print "MODEL 1 RESULTS:"
print model_err_stats
print
print "BIGGEST ERRORS:"
print tw_names_drop

print ".... PLOTTING ERRORS ....\n"
plot_errors(results)


alphas = np.logspace(-4, -.5, 20)
best_alpha = linModel1.cross_validate(alphas=alphas, folds=5)
print "\nFOUND BEST ALPHA TO USE IN MODEL 1"
print ".... TRAINING MODEL 1 with alpha = ...", best_alpha
linModel1 = Model()
linModel1.clean(quantile = .1)
linModel1.train(model = Ridge, perc_train = .9, alpha = best_alpha)
print "\n"
print ".... GETTING RESULTS FOR MODEL 1...."
score = linModel1.r_score
print "ADJUSTED R2 = ", score
print "\n"
results = linModel1.get_results()
print "Model Coefficients:"
coefs = linModel1.get_coefs()
print coefs




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

model_err_stats2 = linModel2.model_results.groupby(['page_id','tw_name'])['perc_diff'].agg(['count','sum','mean']).sort_values('mean',ascending = False, axis =0).reset_index()
model_err_stats2.rename(columns = {'count':'frequency', 'sum':'err_sum' ,'mean':'err_mean'}, inplace = True)
tw_names_drop2 = model_err_stats2[model_err_stats2.err_mean > err_threshold][['page_id','tw_name']].drop_duplicates()

print "MODEL 2 RESULTS:"
print model_err_stats2
print
print "BIGGEST ERRORS:"
print tw_names_drop2

print ".... PLOTTING ERRORS FOR MODEL 2...."
plot_errors(results2)



print "*" * 30
print "BUILDING MODEL FOR TWITTER USERS GUILTY OF LARGEST ERRORS"

linModel3 = Model()
linModel3.tw_data = linModel3.tw_data[linModel3.tw_data.page_id.isin(tw_names_drop.page_id)]
linModel3.clean(quantile = .1)
best_alpha_3 = linModel3.cross_validate(alphas=alphas, folds=10)
linModel3.train(model = Ridge, perc_train = .9, alpha = best_alpha_3)
score3 = linModel3.r_score
results3 = linModel3.get_results()
print results3
model_err_stats3 = linModel3.model_results.groupby(['page_id','tw_name'])['perc_diff'].agg(['count','sum','mean']).sort_values('mean',ascending = False, axis =0).reset_index()
model_err_stats3.rename(columns = {'count':'frequency', 'sum':'err_sum' ,'mean':'err_mean'}, inplace = True)

print "MODEL 3 RESULTS:"
print model_err_stats3

print ".... PLOTTING ERRORS FOR MODEL 3...."
plot_errors(results3)




