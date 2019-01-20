[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **RVOLharfnn** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml


Name of Quantlet: RVOLharfnn

Published in: Master thesis

Description: 'Forecasts the realized volatility of a portfolio from cryptocurrencies by the means 
    	      of the Heterogeneous autoregressive (HAR) model and a hybrid between HAR and a simple
	      feedforward neural network (FNN-HAR).'

Keywords: 'cryptocurrency, RV, HAR, FNN, realized, volatility'

Author: Ivan Mitkov

Submitted: Fri, January 11 2019 by Ivan Mitkov

Datafile: raw_data.csv

Output: 
- harfnn_errors_high_vol_long.csv, 
- harfnn_errors_high_vol_short.csv, 
- harfnn_errors_low_vol_long.csv, 
- harfnn_errors_low_vol_short.csv, 
- harfnn_outofsample_predictions_high_vol_long.csv,
- harfnn_outofsample_predictions_high_vol_short.csv,
- harfnn_outofsample_predictions_low_vol_long.csv,
- harfnn_outofsample_predictions_low_vol_short.csv
```

### PYTHON Code
```python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def train_test_data(dataframe, fraction, RNN):
    import pandas as pd
    import numpy as np
    from numpy import sqrt

    dataframe = dataframe.reset_index(drop = True)
    
    rows = np.quantile((np.array(dataframe.index)), fraction).astype(int)+1
    
    train_X = dataframe[:rows]
    train_X = train_X[['DAILY_RV', 'WEEKLY_RV', 'MONTHLY_RV']]
    train_X.reset_index(drop = True, inplace = True)
    
    test_X = dataframe[rows:]
    test_X = test_X[['DAILY_RV', 'WEEKLY_RV', 'MONTHLY_RV']]
    test_X.reset_index(drop = True, inplace = True)
    
    # Split the targets into training/testing sets
    train_Y = dataframe['TARGET_RV'][:rows]
    test_Y = dataframe['TARGET_RV'][rows:]
    if RNN == False:
        test_X.reset_index(drop = True, inplace = True)
        test_Y.reset_index(drop = True, inplace = True)
        return(train_X, train_Y, test_X, test_Y)
    else:
        train_X = train_X.values[..., 0]
        test_X = test_X.values[..., 0]
        train_X = np.reshape(train_X, (train_X.shape[0], 1, 1))
        test_X = np.reshape(test_X, (test_X.shape[0], 1, 1))
        test_X.reset_index(drop = True, inplace = True)
        test_Y.reset_index(drop = True, inplace = True)
        return(train_X, train_Y, test_X, test_Y) 
        
def train_test_var(dataframe, n_rows, RNN):
    import pandas as pd
    import numpy as np
    from numpy import sqrt

    dataframe = dataframe.reset_index(drop = True)
    
    rows = n_rows
    
    train_X = dataframe[:rows]
    train_X = train_X[['DAILY_RV', 'WEEKLY_RV', 'MONTHLY_RV']]
    train_X.reset_index(drop = True, inplace = True)
    
    test_X = dataframe[rows:]
    test_X = test_X[['DAILY_RV', 'WEEKLY_RV', 'MONTHLY_RV']]
    test_X.reset_index(drop = True, inplace = True)
    
    # Split the targets into training/testing sets
    train_Y = dataframe['TARGET_RV'][:rows]
    test_Y = dataframe['TARGET_RV'][rows:]
    if RNN == False:
        return(train_X, train_Y, test_X, test_Y)
    else:
        train_X = train_X.values[..., 0]
        test_X = test_X.values[..., 0]
        train_X = np.reshape(train_X, (train_X.shape[0], 1, 1))
        test_X = np.reshape(test_X, (test_X.shape[0], 1, 1))
        return(train_X, train_Y, test_X, test_Y)         

def naive_erros(train_Y, test_Y, NAIVE_train, NAIVE_test):
    import pandas as pd
    from math import sqrt
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error 
    print('\nTest errors HAR-RV: \n\nRoot mean squared error: ', "{:.2e}".format(sqrt(mean_squared_error(train_Y, NAIVE_train))),
                          '\nMean absolute error: ', "{:.2e}".format(mean_absolute_error(train_Y, NAIVE_train)),
                          '\nMean squared logarithmic error: ', "{:.2e}".format(mean_squared_log_error(train_Y, NAIVE_train)))
    
    print('\nTest errors HAR-RV: \n\nRoot mean squared error: ', "{:.2e}".format(sqrt(mean_squared_error(test_Y, NAIVE_test))),
                          '\nMean absolute error: ', "{:.2e}".format(mean_absolute_error(test_Y, NAIVE_test)),
                          '\nMean squared logarithmic error: ', "{:.2e}".format(mean_squared_log_error(test_Y, NAIVE_test)))
    error = pd.DataFrame()
    error['COEF'] = pd.Series(('Root mean squared error', 'Mean absolute error', 'Mean squared logarithmic error'))
    error['COEF'] = pd.Series(('Root mean squared error', 'Mean absolute error', 'Mean squared logarithmic error', 'Root mean squared error', 'Mean absolute error', 'Mean squared logarithmic error'))
    error['MODEL'] = 'Naive'
    error['TRAIN'] = pd.Series((sqrt(mean_squared_error(train_Y, NAIVE_train)),
                                 mean_absolute_error(train_Y, NAIVE_train),
                                 mean_squared_log_error(train_Y, NAIVE_train)))
        
    error['TEST'] = pd.Series((sqrt(mean_squared_error(test_Y, NAIVE_test)),
                                 mean_absolute_error(test_Y, NAIVE_test),
                                 mean_squared_log_error(test_Y, NAIVE_test)))

    error['ERRORS'] = pd.Series((sqrt(mean_squared_error(train_Y, NAIVE_train)),
                                 mean_absolute_error(train_Y, NAIVE_train),
                                 mean_squared_log_error(train_Y, NAIVE_train),
                                 sqrt(mean_squared_error(test_Y, NAIVE_test)),
                                 mean_absolute_error(test_Y, NAIVE_test),
                                 mean_squared_log_error(test_Y, NAIVE_test)))                                 

    return(error)
    
    
        
class har_fnn(object):
    
    def __init__(self, RV_type, trainX, trainY, testX, testY):       
        self.RV_type = RV_type
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

    def train(self):
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn import linear_model
        from keras.models import Sequential
        from keras.layers import Dense
        from keras import optimizers
        
        if self.RV_type == 'HAR-RV':

            # Create linear regression object
            self.regr = linear_model.LinearRegression(fit_intercept = True)
  
            # Train the model using the training sets
            self.regr.fit(self.trainX, self.trainY)
            print('The model was successfully trained')
        else:
            self.model = Sequential()
            self.model.add(Dense(2, input_dim = 3, activation='elu'))
            self.model.add(Dense(1))
            adam = optimizers.adam(lr = 0.05)
            self.model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
            self.model.fit(self.trainX.values, self.trainY.values, epochs = 100, batch_size = 500, verbose = 0, shuffle = False)
            print(self.model.summary())
            
    def predict(self):
        import pandas as pd
        if self.RV_type == 'HAR-RV':
            
            # Make predictions using the testing set
            self.y_train_pred = self.regr.predict(self.trainX)
            self.y_test_pred = self.regr.predict(self.testX)
            return(pd.Series(self.y_train_pred), pd.Series(self.y_test_pred))            
        else:
            self.y_train_pred = self.model.predict(self.trainX)
            self.y_test_pred = self.model.predict(self.testX)
            return(pd.Series(self.y_train_pred[:,0]), pd.Series(self.y_test_pred[:,0]))

    
    def evaluate(self, train = False, test = False):
        import pandas as pd
        from math import sqrt
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error 
        
        if train == True:
            if self.RV_type == 'HAR-RV':
                # Some Error Measures
                print('\nTrain errors HAR-RV: \n\nRoot mean squared error: ', "{:.2e}".format(sqrt(mean_squared_error(self.trainY, self.y_train_pred))),
                      '\nMean absolute error: ', "{:.2e}".format(mean_absolute_error(self.trainY, self.y_train_pred)),
                      '\nMean squared logarithmic error: ', "{:.2e}".format(mean_squared_log_error(self.trainY, self.y_train_pred)))
            else:
                #print(self.model.evaluate(self.trainX, self.trainY, verbose=0))
                print("\n\n Train data FNN-HAR: \n\nRoot mean squared error: ", "{:.2e}".format(sqrt(mean_squared_error(self.trainY, self.y_train_pred))),
                      '\nMean absolute error: ', "{:.2e}".format(mean_absolute_error(self.trainY, self.y_train_pred)),
                      '\nMean squared logarithmic error: ', "{:.2e}".format(mean_squared_log_error(self.trainY, self.y_train_pred)))
                
        if test == True:
            if self.RV_type == 'HAR-RV':
                # Some Error Measures
                print('\nTest errors HAR-RV: \n\nRoot mean squared error: ', "{:.2e}".format(sqrt(mean_squared_error(self.testY, self.y_test_pred))),
                      '\nMean absolute error: ', "{:.2e}".format(mean_absolute_error(self.testY, self.y_test_pred)),
                      '\nMean squared logarithmic error: ', "{:.2e}".format(mean_squared_log_error(self.testY, self.y_test_pred)))
            else:
                print('\n\nTest data FNN-HAR:',  '\n\nRoot mean squared error: ', "{:.2e}".format(sqrt(mean_squared_error(self.testY, self.y_test_pred))),
                      '\nMean absolute error: ', "{:.2e}".format(mean_absolute_error(self.testY, self.y_test_pred)),
                      '\nMean squared logarithmic error: ', "{:.2e}".format(mean_squared_log_error(self.testY, self.y_test_pred)))
        
                                
    
    def error_df(self):
        import pandas as pd
        from math import sqrt
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error 

        error = pd.DataFrame()
        error['COEF'] = pd.Series(('Root mean squared error', 'Mean absolute error', 'Mean squared logarithmic error'))
        error['MODEL'] = self.RV_type

        error['TRAIN'] = pd.Series((sqrt(mean_squared_error(self.trainY, self.y_train_pred)),
                                         mean_absolute_error(self.trainY, self.y_train_pred),
                                         mean_squared_log_error(self.trainY, self.y_train_pred)))
        
        error['TEST'] = pd.Series((sqrt(mean_squared_error(self.testY, self.y_test_pred)),
                                        mean_absolute_error(self.testY, self.y_test_pred),
                                        mean_squared_log_error(self.testY, self.y_test_pred)))
        
        error['ERRORS'] = pd.Series((sqrt(mean_squared_error(self.trainY, self.y_train_pred)),
                                 mean_absolute_error(self.trainY, self.y_train_pred),
                                 mean_squared_log_error(self.trainY, self.y_train_pred),
                                 sqrt(mean_squared_error(self.testY, self.y_test_pred)),
                                 mean_absolute_error(self.testY, self.y_test_pred),
                                 mean_squared_log_error(self.testY, self.y_test_pred)))                                 

        return(error)
        

# Actual analysis
        
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set wordking directiory
import os

# Please note that the code works without any errors only if you run the whole script at once
os.chdir(os.path.dirname(__file__))

# If you run partial codes then you have to replace manually the working directory with your path
# os.chdir(r'/hier/comes/your/path/link')

# Impot of packages
from dalib.packages import *

# In order to keep constant outputs, avoiding different weights initialization
from numpy.random import seed
seed(1)

# Slicing the data frame

"""
Please note that for at our empirical work we consider four different time perios as follows:
    1. High volatile times:
        1.1 start_date = '2017-10-01'; end_date = '2018-01-01' # long version
        1.2 start_date = '2017-11-01'; end_date = '2017-12-01' # short version        
    2. Low volatile times:
        2.1 start_date = '2018-04-15'; end_date = '2018-07-15' # long version
        2.2 start_date = '2018-05-15'; end_date = '2017-08-15' # short version                

"""
dates = [('2017-10-01', '2018-01-01', 'high_vol_long'), ('2017-11-01', '2017-12-01', 'high_vol_short'), 
         ('2018-04-15', '2018-07-15', 'low_vol_long'), ('2018-05-15', '2018-06-15', 'low_vol_short')]


# # Scenario high volatiolity time, longer training
Scenario = [dates[0][0], dates[0][1], dates[0][2]] 

# Import data
df = pd.read_csv(r'data/raw_data.csv', sep = ';')
df = df[(df['DATE'] > Scenario[0]) & (df['DATE'] < Scenario[1])].reset_index(drop = True)

# Train-Test-Data
train_X, train_Y, test_X, test_Y = train_test_data(dataframe = df, 
                                                   fraction = 0.8,
                                                   RNN = False)

# Naïve model
NAIVE_train = df['NAIVE'][(test_Y.shape[0]):].reset_index(drop = True)
NAIVE_test = df['NAIVE'].tail(test_Y.shape[0]).reset_index(drop = True)
NAIVE_errors = naive_erros(train_Y = train_Y, test_Y = test_Y, NAIVE_train = NAIVE_train, NAIVE_test = NAIVE_test)

# HAR-RV
har = har_fnn('HAR-RV', trainX = train_X, trainY = train_Y, testX = test_X, testY = test_Y)
har.train()
predicted_train_Y_HAR, predicted_test_Y_HAR = har.predict()
har.evaluate(train = True, test = True)
har_errors = har.error_df()

# FNN-HAR
fnn_har = har_fnn('FNN-ANN', trainX = train_X, trainY = train_Y, testX = test_X, testY = test_Y)
fnn_har.train()
predicted_train_Y_FNN, predicted_test_Y_FNN = fnn_har.predict()
fnn_har.evaluate(train = True, test = True)
fnn_har_errors = fnn_har.error_df()


# Save the predictions and errors
errors_naive_har_fnn = NAIVE_errors.append([har_errors, fnn_har_errors]).reset_index(drop = True)
errors_naive_har_fnn.to_csv(r'output/harfnn_errors_' + str(Scenario[2]) + '.csv', sep = ';')

outofsample_predictions = pd.concat([pd.Series(test_Y), NAIVE_test, predicted_test_Y_HAR, predicted_test_Y_FNN], axis = 1).reset_index(drop = True)
outofsample_predictions.columns = ['DAILY_RV', 'NAIVE', 'HAR', 'FNN-HAR']
outofsample_predictions.to_csv(r'output/harfnn_outofsample_predictions_' + str(Scenario[2]) + '.csv', sep = ';')



# # Scenario high volatiolity time, shorter training
Scenario = [dates[1][0], dates[1][1], dates[1][2]] # Scenario high volatiolity time, longer training

# Import data
df = pd.read_csv(r'data/raw_data.csv', sep = ';')
df = df[(df['DATE'] > Scenario[0]) & (df['DATE'] < Scenario[1])].reset_index(drop = True)

# Train-Test-Data
train_X, train_Y, test_X, test_Y = train_test_data(dataframe = df, 
                                                   fraction = 0.8,
                                                   RNN = False)

# Naïve model
NAIVE_train = df['NAIVE'][(test_Y.shape[0]):].reset_index(drop = True)
NAIVE_test = df['NAIVE'].tail(test_Y.shape[0]).reset_index(drop = True)
NAIVE_errors = naive_erros(train_Y = train_Y, test_Y = test_Y, NAIVE_train = NAIVE_train, NAIVE_test = NAIVE_test)

# HAR-RV
har = har_fnn('HAR-RV', trainX = train_X, trainY = train_Y, testX = test_X, testY = test_Y)
har.train()
predicted_train_Y_HAR, predicted_test_Y_HAR = har.predict()
har.evaluate(train = True, test = True)
har_errors = har.error_df()

# FNN-HAR
fnn_har = har_fnn('FNN-ANN', trainX = train_X, trainY = train_Y, testX = test_X, testY = test_Y)
fnn_har.train()
predicted_train_Y_FNN, predicted_test_Y_FNN = fnn_har.predict()
fnn_har.evaluate(train = True, test = True)
fnn_har_errors = fnn_har.error_df()


# Save the predictions and errors
errors_naive_har_fnn = NAIVE_errors.append([har_errors, fnn_har_errors]).reset_index(drop = True)
errors_naive_har_fnn.to_csv(r'output/harfnn_errors_' + str(Scenario[2]) + '.csv', sep = ';')

outofsample_predictions = pd.concat([pd.Series(test_Y), NAIVE_test, predicted_test_Y_HAR, predicted_test_Y_FNN], axis = 1).reset_index(drop = True)
outofsample_predictions.columns = ['DAILY_RV', 'NAIVE', 'HAR', 'FNN-HAR']
outofsample_predictions.to_csv(r'output/harfnn_outofsample_predictions_' + str(Scenario[2]) + '.csv', sep = ';')



# # Scenario low volatiolity time, longer training
Scenario = [dates[2][0], dates[2][1], dates[2][2]] 

# Import data
df = pd.read_csv(r'data/raw_data.csv', sep = ';')
df = df[(df['DATE'] > Scenario[0]) & (df['DATE'] < Scenario[1])].reset_index(drop = True)

# Train-Test-Data
train_X, train_Y, test_X, test_Y = train_test_data(dataframe = df, 
                                                   fraction = 0.8,
                                                   RNN = False)

# Naïve model
NAIVE_train = df['NAIVE'][(test_Y.shape[0]):].reset_index(drop = True)
NAIVE_test = df['NAIVE'].tail(test_Y.shape[0]).reset_index(drop = True)
NAIVE_errors = naive_erros(train_Y = train_Y, test_Y = test_Y, NAIVE_train = NAIVE_train, NAIVE_test = NAIVE_test)

# HAR-RV
har = har_fnn('HAR-RV', trainX = train_X, trainY = train_Y, testX = test_X, testY = test_Y)
har.train()
predicted_train_Y_HAR, predicted_test_Y_HAR = har.predict()
har.evaluate(train = True, test = True)
har_errors = har.error_df()

# FNN-HAR
fnn_har = har_fnn('FNN-ANN', trainX = train_X, trainY = train_Y, testX = test_X, testY = test_Y)
fnn_har.train()
predicted_train_Y_FNN, predicted_test_Y_FNN = fnn_har.predict()
fnn_har.evaluate(train = True, test = True)
fnn_har_errors = fnn_har.error_df()


# Save the predictions and errors
errors_naive_har_fnn = NAIVE_errors.append([har_errors, fnn_har_errors]).reset_index(drop = True)
errors_naive_har_fnn.to_csv(r'output/harfnn_errors_' + str(Scenario[2]) + '.csv', sep = ';')

outofsample_predictions = pd.concat([pd.Series(test_Y), NAIVE_test, predicted_test_Y_HAR, predicted_test_Y_FNN], axis = 1).reset_index(drop = True)
outofsample_predictions.columns = ['DAILY_RV', 'NAIVE', 'HAR', 'FNN-HAR']
outofsample_predictions.to_csv(r'output/harfnn_outofsample_predictions_' + str(Scenario[2]) + '.csv', sep = ';')



# # Scenario low volatiolity time, shorter training
Scenario = [dates[3][0], dates[3][1], dates[3][2]] # Scenario high volatiolity time, longer training

# Import data
df = pd.read_csv(r'data/raw_data.csv', sep = ';')
df = df[(df['DATE'] > Scenario[0]) & (df['DATE'] < Scenario[1])].reset_index(drop = True)

# Train-Test-Data
train_X, train_Y, test_X, test_Y = train_test_data(dataframe = df, 
                                                   fraction = 0.8,
                                                   RNN = False)

# Naïve model
NAIVE_train = df['NAIVE'][(test_Y.shape[0]):].reset_index(drop = True)
NAIVE_test = df['NAIVE'].tail(test_Y.shape[0]).reset_index(drop = True)
NAIVE_errors = naive_erros(train_Y = train_Y, test_Y = test_Y, NAIVE_train = NAIVE_train, NAIVE_test = NAIVE_test)

# HAR-RV
har = har_fnn('HAR-RV', trainX = train_X, trainY = train_Y, testX = test_X, testY = test_Y)
har.train()
predicted_train_Y_HAR, predicted_test_Y_HAR = har.predict()
har.evaluate(train = True, test = True)
har_errors = har.error_df()

# FNN-HAR
fnn_har = har_fnn('FNN-ANN', trainX = train_X, trainY = train_Y, testX = test_X, testY = test_Y)
fnn_har.train()
predicted_train_Y_FNN, predicted_test_Y_FNN = fnn_har.predict()
fnn_har.evaluate(train = True, test = True)
fnn_har_errors = fnn_har.error_df()


# Save the predictions and errors
errors_naive_har_fnn = NAIVE_errors.append([har_errors, fnn_har_errors]).reset_index(drop = True)
errors_naive_har_fnn.to_csv(r'output/harfnn_errors_' + str(Scenario[2]) + '.csv', sep = ';')

outofsample_predictions = pd.concat([pd.Series(test_Y), NAIVE_test, predicted_test_Y_HAR, predicted_test_Y_FNN], axis = 1).reset_index(drop = True)
outofsample_predictions.columns = ['DAILY_RV', 'NAIVE', 'HAR', 'FNN-HAR']
outofsample_predictions.to_csv(r'output/harfnn_outofsample_predictions_' + str(Scenario[2]) + '.csv', sep = ';')
```

automatically created on 2019-01-20