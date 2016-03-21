'''
script for loading data and importing libraries
@author: Gabi
'''


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


data_path = "/Users/Gabi/Documents/Shareablee/Twitter/data/"

# twitter_data_path = os.path.join(data_path, 'twitter_data')
twitter_data_path = os.path.join(data_path, 'twitter_data_102015_012016') # added 102015-122015
user_data_path = os.path.join(data_path, 'tw_account_info')
project_path =  '/Users/Gabi/Documents/Shareablee/Twitter/'
os.chdir(project_path)

pd_read_names = ['index','time', 'impressions', 'retweets', 'page_id', 'twitter_id', 'followers']
pd_read_dtype = [np.str, np.str, np.int, np.int, np.str, np.int, np.int]

pd_read_names_user = ['user_id','tw_id', 'tw_name', 'cat_id', 'cat_name']
pd_read_dtype_user = [np.str, np.str, np.str, np.str, np.str]

pd_kwargs = {
    'header': 0,
    'sep': ',',
    'escapechar': '\\',
    'names': pd_read_names,
    'dtype': dict(zip(pd_read_names, pd_read_dtype)),
    'usecols': ['time', 'impressions', 'retweets', 'page_id', 'followers'],
    'parse_dates' : [0]
}

pd_user_kwargs = {
    'header': 0,
    'sep': ',',
    'escapechar': '\\',
    'names': pd_read_names_user,
    'dtype': dict(zip(pd_read_names_user, pd_read_dtype_user))
    }


# ------------------------------------------------
# read twitter Jan 2016 data
# read twitter user category data
# ------------------------------------------------

'''
Input:
	Arguments for reading Twitter data (Twitter Data - JAN 2016)

Output:
    Dataframe
'''
def load_tw_data(**kwargs):
    twitter_data = pd.read_table(twitter_data_path, **pd_kwargs)
    return twitter_data


'''
Input:
	Arguments for reading Twitter User Account Info

Output:
    Dataframe
'''
def load_user_data(**kwargs):
    user_data = pd.read_table(user_data_path, **pd_user_kwargs)
    return user_data

twitter_data = load_tw_data()
user_data = load_user_data()

tw_samples = user_data[['user_id','tw_name']].drop_duplicates()
tw_samples.rename(columns = {'user_id':'page_id'}, inplace=True)


# twitter_data.merge(tw_samples, how = "inner", left_on = "page_id", right_on = "page_id")
# twitter_data = pd.read_table(data_path, **pd_kwargs)
# user_data = pd.read_table(user_data_path, **pd_user_kwargs)

