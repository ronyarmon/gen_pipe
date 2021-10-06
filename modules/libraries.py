import sys
import time
import os
import re
from functools import reduce
import urllib.request

import pandas as pd
import numpy as np
import tensorflow as tf

import sklearn
from sklearn.impute import SimpleImputer
imputer_strategies = {'m': 'median', 'me': 'mean', 'mf': 'most_frequent', 'c': 'constant'}
from sklearn.model_selection import train_test_split
# Models
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import LinearSVR, SVR


# Evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from scipy.stats import zscore

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

print('python libraries imported')



