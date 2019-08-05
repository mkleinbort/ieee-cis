#####################################
# Setting the path to the directory #
#####################################

import sys
import os

current_dir = os.path.abspath('.')
path_to_src_dir = f'../{current_dir}/'

sys.path.insert(0, path_to_src_dir)


#####################################
#   Core "always used" libraries    #
#####################################

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pandas import datetime

# Prefered Options
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

pd.options.display.max_columns = 200
pd.options.display.float_format = '{:,.2f}'.format


#####################################
#   Code Snipet of Personal Shame   #
#####################################

import warnings
warnings.filterwarnings('ignore')


#####################################
#          Jupyter Flags            #
#####################################

NOTEBOOK_SETUP = '''
display(HTML("<style>.container { width:90% !important; }</style>"))

%config InlineBackend.figure_format = 'retina'
%load_ext line_profiler
%matplotlib inline
'''

#####################################
#   Other good to have functions    #
#####################################

from IPython.display import display, HTML
from glob import glob

try:
    import shap
    shap.initjs()
except:
    print('Note, could not import shap and run shap.initjs()')

try:
    import missingno as msgn
except:
    print('Note, could not import missingno as msgn')

#####################################
#   Common ML functions - helpful   #
#####################################

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score, r2_score, mutual_info_score

#####################################
#         Common modules            #
#####################################

import os
import re

import requests
import time

from functools import reduce, partial
import inspect

#####################################
#         Custom Functions          #
#####################################

try:
    from utils import *
except:
    print('ERROR, could not load * from src.utils')