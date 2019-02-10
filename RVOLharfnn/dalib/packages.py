#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# Make sure that all of the modules are already installed in Anaconda. If not, 
# the following comands should be ran in Terminal
"""
pip install pandas
pip install matplotlib
pip install numpy
pip install statsmodels
pip install scipy
pip install tensorflow
pip install --upgrade tensorflow
pip install keras
pip install --upgrade keras
pip install -U scikit-learn
"""

# Improting the packages
import pandas as pd
from pandas import Series

import matplotlib.pyplot as plt 
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import *

import itertools
import sys
import pylab 
import scipy.stats as stats
from scipy.stats import norm                                                  
from numpy.random import normal
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.layers import *
from keras import optimizers
from keras.constraints import NonNeg
from keras.layers.advanced_activations import *
