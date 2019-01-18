#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 14:39:15 2018

@author: ivanmitkov
"""

# Import the required moduldes 

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy
from numpy.random import seed
seed(1)

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
    train_Y.reset_index(drop = True, inplace = True)
    test_Y = dataframe['TARGET_RV'][rows:]
    test_Y.reset_index(drop = True, inplace = True)
    
    if RNN == False:
        return(train_X, train_Y, test_X, test_Y)
    else:
        train_X = train_X.values[..., 0]
        test_X = test_X.values[..., 0]
        train_X = np.reshape(train_X, (train_X.shape[0], 1, 1))
        test_X = np.reshape(test_X, (test_X.shape[0], 1, 1))
        return(train_X, train_Y, test_X, test_Y) 
    
class Neural_Networks(object):
    import numpy as np
    import math
    from sklearn import metrics
    from keras import optimizers
    import pandas as pd
    import numpy as np
    from keras.layers import SimpleRNN
    from keras.layers import Dense
    from keras.models import Sequential
    from keras import optimizers
    import math
    from sklearn.metrics import mean_squared_error
    from numpy.random import seed
    seed(1)
    
    def __init__(self, NN_type, nn_inputs, train_X, train_Y, test_X, test_Y):       
        self.NN_type = NN_type
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.nn_inputs = nn_inputs 
        self.output = 1 
        self.time_steps = 1

    def nn_grid_params(self, dict_models, learning_rate, n_epochs, batch_size):
        """
        The function makes a grid search over the hyperparameters for a Simple Recurrent Neural Network.
        """
        import pandas as pd
        import numpy as np
        from keras.layers import SimpleRNN
        from keras.layers import LSTM
        from keras.layers import GRU        
        from keras.layers import Dense
        from keras.models import Sequential
        from keras.layers import LeakyReLU
        from keras import optimizers
        import math
        from sklearn.metrics import mean_squared_error
        
        
        if self.NN_type == 'SimpleRNN':
            result = pd.DataFrame()    
            for i in list(dict_models.keys()):
                model = Sequential()
                if dict_models[i]['n_layers'] == 1:
                    model.add(SimpleRNN(dict_models[i]['n_units'][0], input_shape=(self.time_steps, self.nn_inputs), activation = dict_models[i]['activ_funct']))
                else:
                    model.add(SimpleRNN(dict_models[i]['n_units'][0], input_shape=(self.time_steps, self.nn_inputs), return_sequences = True, activation = dict_models[i]['activ_funct']))
                for j in range(2, dict_models[i]['n_layers'] + 1):
                    if j < dict_models[i]['n_layers']:
                        model.add(SimpleRNN(dict_models[i]['n_units'][j-1], return_sequences = True, activation = dict_models[i]['activ_funct']))
                    else:
                        model.add(SimpleRNN(dict_models[i]['n_units'][len(dict_models[i]['n_units'])-1], return_sequences = False, activation = dict_models[i]['activ_funct']))
                
                model.add(Dense(self.output))
                
                for rate in learning_rate:
                    adam = optimizers.adam(lr = rate)
                    model.compile(loss='mse', optimizer='adam')
                    
                    for epoch in n_epochs:
                        for size in batch_size:
                            model.fit(self.train_X, self.train_Y, epochs = epoch, batch_size = size, verbose = 1)
                            grid = pd.DataFrame()
                            grid['MODEL'] = [(i)]
                            grid['LAYERS'] = [(dict_models[i]['n_layers'])]
                            grid['NEURONS'] = [(dict_models[i]['n_units'])]
                            grid['RATE'] = rate
                            grid['SIZE'] = size
                            grid['MSE'] = model.evaluate(self.train_X, self.train_Y, verbose=0)
                            result = result.append(grid).reset_index(drop = True)
            result.sort_values(by=['MSE'], inplace = True)                
            print(result[:5])
            
        if self.NN_type == 'LSTM':
            result = pd.DataFrame()                
            for i in list(dict_models.keys()):
                model = Sequential()
                if dict_models[i]['n_layers'] == 1:
                    model.add(LSTM(dict_models[i]['n_units'][0], input_shape=(self.time_steps, self.nn_inputs), activation = dict_models[i]['activ_funct']))
                else:
                    model.add(LSTM(dict_models[i]['n_units'][0], input_shape=(self.time_steps, self.nn_inputs), return_sequences = True, activation = dict_models[i]['activ_funct']))
                for j in range(2, dict_models[i]['n_layers'] + 1):
                    if j < dict_models[i]['n_layers']:
                        model.add(LSTM(dict_models[i]['n_units'][j-1], return_sequences = True, activation = dict_models[i]['activ_funct']))
                    else:
                        model.add(LSTM(dict_models[i]['n_units'][len(dict_models[i]['n_units'])-1], return_sequences = False, activation = dict_models[i]['activ_funct']))
                
                model.add(Dense(self.output))
                
                for rate in learning_rate:
                    adam = optimizers.adam(lr = rate)
                    model.compile(loss='mse', optimizer='adam')
                    
                    for epoch in n_epochs:
                        for size in batch_size:
                            model.fit(self.train_X, self.train_Y, epochs = epoch, batch_size = size, verbose = 0)
                            grid = pd.DataFrame()
                            grid['MODEL'] = [(i)]
                            grid['LAYERS'] = [(dict_models[i]['n_layers'])]
                            grid['NEURONS'] = [(dict_models[i]['n_units'])]
                            grid['RATE'] = rate
                            grid['SIZE'] = size
                            grid['MSE'] = model.evaluate(self.train_X, self.train_Y, verbose=0)
                            result = result.append(grid).reset_index(drop = True)
            result.sort_values(by=['MSE'], inplace = True)
            print(result[:5])     
        
        if self.NN_type == 'GRU':
            result = pd.DataFrame()
            for i in list(dict_models.keys()):
                model = Sequential()
                if dict_models[i]['n_layers'] == 1:
                    model.add(GRU(dict_models[i]['n_units'][0], input_shape=(self.time_steps, self.nn_inputs), activation = dict_models[i]['activ_funct']))
                else:
                    model.add(GRU(dict_models[i]['n_units'][0], input_shape=(self.time_steps, self.nn_inputs), return_sequences = True, activation = dict_models[i]['activ_funct']))
                for j in range(2, dict_models[i]['n_layers'] + 1):
                    if j < dict_models[i]['n_layers']:
                        model.add(GRU(dict_models[i]['n_units'][j-1], return_sequences = True, activation = dict_models[i]['activ_funct']))
                    else:
                        model.add(GRU(dict_models[i]['n_units'][len(dict_models[i]['n_units'])-1], return_sequences = False, activation = dict_models[i]['activ_funct']))
                
                model.add(Dense(self.output))
                
                for rate in learning_rate:
                    adam = optimizers.adam(lr = rate)
                    model.compile(loss='mse', optimizer='adam')
                    
                    for epoch in n_epochs:
                        for size in batch_size:
                            model.fit(self.train_X, self.train_Y, epochs = epoch, batch_size = size, verbose = 0)
                            grid = pd.DataFrame()
                            grid['MODEL'] = [(i)]
                            grid['LAYERS'] = [(dict_models[i]['n_layers'])]
                            grid['NEURONS'] = [(dict_models[i]['n_units'])]
                            grid['RATE'] = rate
                            grid['SIZE'] = size
                            grid['MSE'] = model.evaluate(self.train_X, self.train_Y, verbose=0)
                            result = result.append(grid).reset_index(drop = True)
            result.sort_values(by=['MSE'], inplace = True)                
            print(result[:5])
        
    def create_nn(self, n_layers, n_units, n_epochs, batch_size, activ_funct, learning_rate):
        from keras import optimizers
        #K.clear_session()
        self.model = Sequential()
        """
        The function trains a simple Recurrent neural network.
        
        Parameters:
            time_steps: Sequence of time steps back, that have to be included in the network.
            
            n_units: Number of neurons in the corresponding layer. Input as a list, eg [neurons_1_layer, neurons_2_layer]
            
            n_layers: Number of layers. 
            
            self.nn_inputs: Number inputs. Since we are concentrated just on one variable, it is one for us.
            
            n_classes: Number of outputs. Since we are concentrated just on one variable, it is one for us.
            
            act: Activation function as a string.
            
        """
        from keras.layers import SimpleRNN
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import GRU
        from keras.layers.advanced_activations import LeakyReLU, PReLU
        import math
        
        if self.NN_type == 'SimpleRNN':

            if n_layers == 1:
                self.model.add(SimpleRNN(n_units[0], input_shape=(self.time_steps, self.nn_inputs), activation = activ_funct))
            else:
                self.model.add(SimpleRNN(n_units[0], input_shape=(self.time_steps, self.nn_inputs), return_sequences = True, activation = activ_funct))
            for i in range(2, n_layers+1):
                if i < n_layers:
                    self.model.add(SimpleRNN(n_units[i-1], return_sequences = True, activation = activ_funct))
                else:
                    self.model.add(SimpleRNN(n_units[len(n_units)-1], return_sequences = False, activation = activ_funct))

        if self.NN_type == 'LSTM':

            if n_layers == 1:
                self.model.add(LSTM(n_units[0], input_shape=(self.time_steps, self.nn_inputs), activation = activ_funct))
            else:
                self.model.add(LSTM(n_units[0], input_shape=(self.time_steps, self.nn_inputs), return_sequences = True, activation = activ_funct))
            for i in range(2, n_layers+1):
                if i < n_layers:
                    self.model.add(LSTM(n_units[i-1], return_sequences = True, activation = activ_funct))
                else:
                    self.model.add(LSTM(n_units[len(n_units)-1], return_sequences = False, activation = activ_funct))

                
        if self.NN_type == 'GRU':

            if n_layers == 1:
                self.model.add(GRU(n_units[0], input_shape=(self.time_steps, self.nn_inputs), activation = activ_funct))
            else:
                self.model.add(GRU(n_units[0], input_shape=(self.time_steps, self.nn_inputs), return_sequences = True, activation = activ_funct))
            for i in range(2, n_layers+1):
                if i < n_layers:
                    self.model.add(GRU(n_units[i-1], return_sequences = True, activation = activ_funct))
                else:
                    self.model.add(GRU(n_units[len(n_units)-1], return_sequences = False, activation = activ_funct))
        self.model.add(Dense(self.output))
        adam = optimizers.adam(lr = learning_rate)
        self.model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse', 'mae', 'mape'])
        self.model.fit(self.train_X, self.train_Y, epochs = n_epochs, batch_size = batch_size, 
                       verbose=0, shuffle = False, validation_data = (self.test_X, self.test_Y))

    def prediction(self):
        import pandas as pd
        self.trainPredict = self.model.predict(self.train_X)
        self.testPredict = self.model.predict(self.test_X)
        return(pd.Series(self.trainPredict[:,0]), pd.Series(self.testPredict[:,0]))
            
    def evaluation(self):
        from math import sqrt
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error          
        print('\n\n Train data', self.NN_type, ': \n\nRoot mean squared error: ', "{:.2e}".format(sqrt(mean_squared_error(self.train_Y, self.trainPredict))),
              '\nMean absolute error: ', "{:.2e}".format(mean_absolute_error(self.train_Y, self.trainPredict)),
              '\nMean squared logarithmic error: ', "{:.2e}".format(mean_squared_log_error(self.train_Y, self.trainPredict)))

        print('\n\n Test data', self.NN_type, ': \n\nRoot mean squared error: ', "{:.2e}".format(sqrt(mean_squared_error(self.test_Y, self.testPredict))),
              '\nMean absolute error: ', "{:.2e}".format(mean_absolute_error(self.test_Y, self.testPredict)),
              '\nMean squared logarithmic error: ', "{:.2e}".format(mean_squared_log_error(self.test_Y, self.testPredict)))
        
    def error_df(self):
        import pandas as pd
        from math import sqrt
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error 

        error = pd.DataFrame()
        error['COEF'] = pd.Series(('Root mean squared error', 'Mean absolute error', 'Mean squared logarithmic error'))
        error['MODEL'] = self.NN_type
        error['TRAIN'] = pd.Series((sqrt(mean_squared_error(self.train_Y, self.trainPredict)),
                                         mean_absolute_error(self.train_Y, self.trainPredict),
                                         mean_squared_log_error(self.train_Y, self.trainPredict)))
        
        error['TEST'] = pd.Series((sqrt(mean_squared_error(self.test_Y, self.testPredict)),
                                        mean_absolute_error(self.test_Y, self.testPredict),
                                        mean_squared_log_error(self.test_Y, self.testPredict)))       
        
        error['ERRORS'] = pd.Series((sqrt(mean_squared_error(self.train_Y, self.trainPredict)),
                                 mean_absolute_error(self.train_Y, self.trainPredict),
                                 mean_squared_log_error(self.train_Y, self.trainPredict),
                                 sqrt(mean_squared_error(self.test_Y, self.testPredict)),
                                 mean_absolute_error(self.test_Y, self.testPredict),
                                 mean_squared_log_error(self.test_Y, self.testPredict)))   
        return(error)
                
    def visualization(self, test_visualization = False):
        import matplotlib.pyplot as plt
        import pandas as pd
        if test_visualization == False:
            trainPredict = self.model.predict(self.train_X)
            plt.figure(dpi = 100)
            plt.plot(self.train_Y, label = 'Target values')
            plt.plot(trainPredict, label = 'Predicted values')
            plt.ylabel('Conditional variance')
            plt.xlabel('Epochs')
            plt.title('Train data: Prediction through ' + self.NN_type)
            plt.legend()
            plt.show()
        else:
            testPredict = self.model.predict(self.test_X)
            plt.figure(dpi = 100)
            plt.plot(pd.Series(self.test_Y), label = 'Target values')
            plt.plot(testPredict, label = 'Predicted values')
            plt.ylabel('Conditional variance')
            plt.xlabel('Epochs')
            plt.title('Test data: Prediction through ' + self.NN_type)
            plt.legend()
            plt.show()                            

# Actual analysis                        

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Please note that the code works without any errors only if you run the whole script at once
os.chdir(os.path.dirname(__file__))

# If you run partial codes then you have to replace manually the working directory with your path
# os.chdir(r'/hier/comes/your/path/link')

# Import of packages
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

# Choose which scenario
Scenario = [dates[0][0], dates[0][1], dates[0][2]] # Scenario high volatiolity time, longer training
Scenario = [dates[1][0], dates[1][1], dates[1][2]] # Scenario high volatiolity time, short training
Scenario = [dates[2][0], dates[2][1], dates[2][2]] # Scenario low volatiolity time, longer training
Scenario = [dates[3][0], dates[3][1], dates[3][2]] # Scenario low volatiolity time, shor training

# Import data
df = pd.read_csv(r'data/raw_data.csv', sep = ';')
df = df[(df['DATE'] > Scenario[0]) & (df['DATE'] < Scenario[1])].reset_index(drop = True)

# Train-Test-Data
train_X, train_Y, test_X, test_Y = train_test_data(dataframe = df, 
                                                   fraction = 0.8,
                                                   RNN = True)

# Simple Recurrent Neural Network
simple_rnn = Neural_Networks('SimpleRNN', nn_inputs = 1, train_X = train_X, train_Y = train_Y,
                             test_X = test_X, test_Y = test_Y)
    
# Train
simple_rnn.create_nn(n_layers = 1, n_units = [2], n_epochs = 100, batch_size = 500, 
                     activ_funct = 'elu', learning_rate = 0.005)
  
# Predictions
rnn_train_prediction, rnn_test_prediction = simple_rnn.prediction()
    
# Evaluation
simple_rnn.evaluation()
    
# Save loss errors
simple_rnn_errors = simple_rnn.error_df()
    
# Visualization
simple_rnn.visualization(test_visualization = False)
simple_rnn.visualization(test_visualization = True)

    
# LSTM Recurrent Neural Network 
lstm = Neural_Networks('LSTM', nn_inputs = 1, train_X = train_X, train_Y = train_Y,
                       test_X = test_X, test_Y = test_Y)
    
# Train
lstm.create_nn(n_layers = 1, n_units = [2], n_epochs = 100, batch_size = 500, 
               activ_funct = 'elu', learning_rate = 0.005)
    
# Predictions
lstm_train_prediction, lstm_test_prediction = lstm.prediction()
    
# Evaluation
lstm.evaluation()
    
# Save loss errors
lstm_errors = lstm.error_df()
    
# Visualization
lstm.visualization(test_visualization = False)
lstm.visualization(test_visualization = True)

# 3. Train the GRU Recurrent Neural Network with the optimal parameters
gru = Neural_Networks('GRU', nn_inputs = 1, train_X = train_X, train_Y = train_Y,
                                 test_X = test_X, test_Y = test_Y)
    
# Train
gru.create_nn(n_layers = 1, n_units = [2], n_epochs = 100, batch_size = 500, 
              activ_funct = 'elu', learning_rate = 0.005)
    
# Predictions
gru_train_prediction, gru_test_prediction = gru.prediction()
    
# Evaluation
gru.evaluation()
    
# Save loss errors
gru_errors = gru.error_df()
    
# Visualization
gru.visualization(test_visualization = False)
gru.visualization(test_visualization = True)
    
# Save the predictions and errors    
errors = simple_rnn_errors.append([lstm_errors, gru_errors]).reset_index(drop = True)
errors.to_csv(r'output/rnn_errors_' + str(Scenario[2]) + '.csv', sep = ';')

outofsample_predictions = pd.concat([pd.Series(test_Y), rnn_test_prediction, lstm_test_prediction, gru_test_prediction], axis = 1).reset_index(drop = True)
outofsample_predictions.columns = ['DAILY_RV', 'SRN', 'LSTM', 'GRU']
outofsample_predictions.to_csv(r'output/rnn_outofsample_predictions_' + str(Scenario[2]) + '.csv', sep = ';')