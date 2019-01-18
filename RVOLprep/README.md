[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **RVOLprep** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml


Name of Quantlet: RVOLprep

Published in: Master thesis

Description: imports the raw data from six cryptocurrencies, builds a price weighted portfolio from all of them and calculated log returns and realized volatility. Additionally, it gives more information about the data with some graphs and descriptive statitstics. 

Keywords: cryptocurrency, returns, RV, realized, volatility, price, weighted, portfolio

Author: Ivan Mitkov

Submitted:  Fri, January 11 2019 by Ivan Mitkov

Datafile: rnn_outofsample_predictions_high_vol_long.csv, rnn_outofsample_predictions_high_vol_short.csv, rnn_outofsample_predictions_low_vol_long.csv, rnn_outofsample_predictions_low_vol_short.csv, harfnn_outofsample_predictions_high_vol_long.csv, harfnn_outofsample_predictions_high_vol_short.csv, harfnn_outofsample_predictions_low_vol_long.csv, harfnn_outofsample_predictions_low_vol_short.csv

Output: high_volatility_long.png, high_volatility_short.png, low_volatility_long.png, low_volatility_short.png

```

### PYTHON Code
```python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 12:18:52 2018

@author: ivanmitkov
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 18:36:59 2018

@author: ivanmitkov
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import of packages, set wordking directiory
import os
os.chdir(r'/Users/ivanmitkov/Documents/Masterarbeit/Skripte')
from dalib.packages import *
from dalib.har_fnn import *
import pickle
data = pickle.load(open('data/price_btc_eth_ltc_str_xmr_xrp_20160101_20180831.p', "rb"))
#data = pickle.load(open(r'/Users/ivanmitkov/Downloads/price_btc_eth_ltc_str_xmr_xrp_20160101_20180831.p', "rb"))
data = data[data.index > '2017-07-15 23:55:00']

#Graph
plt.figure(dpi = 100)
plt.plot(data[('btc_usdt', 'open')].iloc[::288], color = 'b', linewidth=1)
plt.plot(data[('eth_usdt', 'open')].iloc[::288], color = 'r', linewidth=1)
plt.plot(data[('ltc_usdt', 'open')].iloc[::288], color = 'g', linewidth=1)
plt.plot(data[('str_usdt', 'open')].iloc[::288], color = 'orange', linewidth=1)
plt.plot(data[('xmr_usdt', 'open')].iloc[::288], color = 'violet', linewidth=1)
plt.plot(data[('xrp_usdt', 'open')].iloc[::288], color = 'black', linewidth=1)
plt.ylabel('Price')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.title('Prices of the six cryptocurrencies')
plt.show()


# Descriptive statistics
import numpy as np
coins = [data[('btc_usdt', 'open')], data[('eth_usdt', 'open')], data[('ltc_usdt', 'open')], data[('str_usdt', 'open')], data[('xmr_usdt', 'open')], data[('xrp_usdt', 'open')]]
for i in coins:
    print("{:.2e}".format(np.percentile(i, 75)), end=' & ', flush=True)
            
            
# Calculating portfolio
data['PORTFOLIO'] = data[('btc_usdt', 'open')] * (1/6) + \
             data[('eth_usdt', 'open')] *  (1/6) + \
             data[('ltc_usdt', 'open')] *  (1/6) + \
             data[('str_usdt', 'open')] *  (1/6) + \
             data[('xmr_usdt', 'open')] *  (1/6) + \
             data[('xrp_usdt', 'open')] *  (1/6)

# Visualization portfolio
plt.figure(dpi = 100)
plt.plot(data['PORTFOLIO'].iloc[::288], color = 'b', linewidth=1)
plt.ylabel('Price')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.title('Price of the portfolio')
plt.show() 
             
# Log returns
data['RETURNS'] = np.log(data['PORTFOLIO']) - np.log(data['PORTFOLIO'].shift(1))
data['btc_usdt_RETURNS'] = np.log(data[('btc_usdt', 'open')]) - np.log(data[('btc_usdt', 'open')].shift(1))
data['eth_usdt_RETURNS'] = np.log(data[('eth_usdt', 'open')]) - np.log(data[('eth_usdt', 'open')].shift(1))
data['ltc_usdt_RETURNS'] = np.log(data[('ltc_usdt', 'open')]) - np.log(data[('ltc_usdt', 'open')].shift(1))
data['str_usdt_RETURNS'] = np.log(data[('str_usdt', 'open')]) - np.log(data[('str_usdt', 'open')].shift(1))
data['xmr_usdt_RETURNS'] = np.log(data[('xmr_usdt', 'open')]) - np.log(data[('xmr_usdt', 'open')].shift(1))
data['xrp_usdt_RETURNS'] = np.log(data[('xrp_usdt', 'open')]) - np.log(data[('xrp_usdt', 'open')].shift(1))

coins = [data['btc_usdt_RETURNS'], data['eth_usdt_RETURNS'], data['ltc_usdt_RETURNS'], data['str_usdt_RETURNS'], data['xmr_usdt_RETURNS'], data['xrp_usdt_RETURNS'], data['RETURNS']]
for i in coins:
    print("{:.2e}".format(i.std()), end=' & ', flush=True)

from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import stats

data = data[1:]    
















# Log-Returns
data['DATE'] = data.index
data.reset_index(drop = True, inplace = True)

def log_returns(series, dateseries):
    from numpy import sqrt    
        
    """
    Params:
        series: Series to be log transformed.
        
        dataframe: a data frame to which the log returns have to be appended.
    """
    series = pd.Series(series).astype('float64')
    result = pd.DataFrame()
    result['DATE'] = dateseries
    result['PRICE'] = series
    result['RETURNS'] = np.log(result['PRICE']) - np.log(result['PRICE'].shift(1))
    result['DAILY_RETURNS'] = np.log(result['PRICE']) - np.log(result['PRICE'].shift(12 * 24))
    result['SQUARED_RETURNS'] = result['RETURNS'] ** 2
    
    
    result['DAILY_RV'] = sqrt(result['SQUARED_RETURNS'].rolling(12 * 24).sum())
    result['WEEKLY_RV'] = result['DAILY_RV'].rolling(7 * 24 * 12).mean()   
    result['MONTHLY_RV'] = result['DAILY_RV'].rolling(30 * 24 * 12).mean()
    result['TARGET_RV'] = result['DAILY_RV'].shift(-12 * 24)
    result['NAIVE'] = result['DAILY_RV']
    result.dropna(axis=0, how='any', inplace = True)
    result = result[1:]
    result.reset_index(drop = True, inplace = True)
    return(result)
    
# Apply log returns on both data frames    
result = log_returns(series = data['PORTFOLIO'], dateseries = data['DATE'])

plt.figure()
plt.plot(result['NAIVE'])
plt.plot(result['TARGET_RV'])
plt.show()

# Log return visualization
plt.figure(dpi = 100)
plt.plot(data['DATE'], data['RETURNS'], color = 'b', linewidth=1)
plt.ylabel('Price')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.title('Log returns (5 mins frequency) from the portfolio')
plt.show() 
             
# Descriptive statistics             
# mean, deviation, skewness, kurtosis, Jarque-Bera
from scipy.stats import kurtosis
from scipy.stats import skew



             
# Portfolio price vizualization
result.to_csv(r'data/raw_data.csv', index = False, sep = ';')
```

automatically created on 2019-01-18