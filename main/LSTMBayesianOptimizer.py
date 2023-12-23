from kerastuner.tuners import BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import numpy as np

class BayesianLSTMOptimizer:
    def __init__(self, X_train, y_train, backcandles, max_epochs=30, num_initial_points=5):
        self.X_train = X_train
        self.y_train = y_train
        self.backcandles = backcandles
        self.max_epochs = max_epochs
        self.num_initial_points = num_initial_points
        self.best_hps = None

    def build_lstm(self, hp):
        lstm_input = Input(shape=(self.backcandles,15), name='lstm_input')
        inputs = LSTM(
            units=hp.Int('units', min_value=50, max_value=200, step=50),
            activation=hp.Choice('activation', values=['relu', 'tanh']),
            name='first_layer'
        )(lstm_input)

        inputs = Dense(1, name='dense_layer')(inputs)
        output = Activation('linear', name='output')(inputs)
        
        model = Model(inputs=lstm_input, outputs=output)

        adam = optimizers.legacy.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        )
        model.compile(optimizer=adam, loss='mse',metrics=['accuracy'])
        return model

    def optimize(self):
        tuner = BayesianOptimization(
            self.build_lstm,
            objective='val_loss',
            max_trials=3,
            num_initial_points=self.num_initial_points,
            directory='bayesian_optimization',
            project_name='lstm132'
        )

        tuner.search(self.X_train, self.y_train, epochs=self.max_epochs,
                     validation_split=0.1, batch_size=15)

        self.best_hps = tuner.oracle.get_best_trials(1)[0].hyperparameters
        
        best_trial = tuner.oracle.get_best_trials(1)[0]
        #self.best_model = self.build_lstm(best_trial.hyperparameters)
class LSTMModel:
    def __init__(self, input_shape, lstm_units, learning_rate,activation):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.activation=activation
        self.model = self.build_model()

    def build_model(self):
        lstm_input = Input(shape=self.input_shape, name='lstm_input')
        lstm_layer = LSTM(self.lstm_units, activation=self.activation, name='first_layer')(lstm_input)
        dense_layer = Dense(1, name='dense_layer')(lstm_layer)
        output = Activation('sigmoid', name='output')(dense_layer)

        model = Model(inputs=lstm_input, outputs=output)

        adam = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=adam, loss='mse')

        return model

    def train_model(self, X_train, y_train, batch_size=15, epochs=30, validation_split=0.1):
        history = self.model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=validation_split)
        return history

