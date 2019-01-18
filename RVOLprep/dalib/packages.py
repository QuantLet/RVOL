#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:30:57 2018

@author: ivanmitkov
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import jarque_bera
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import adfuller, kpss
import itertools
import sys
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import *
import pylab 
import scipy.stats as stats
from scipy.stats import norm                                                  
from numpy.random import normal
from statsmodels.stats.diagnostic import *
from pyramid.arima import auto_arima
from arch import arch_model

# Neural Netowrks
import math
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
from keras.constraints import NonNeg
from keras.layers.advanced_activations import *
