import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential, Model
from keras.layers import Dense
from keras import layers
import time
import timeit
from keras import optimizers
from tensorflow.keras.layers import MultiHeadAttention
from kerastuner.tuners import BayesianOptimization
from tensorflow.keras.layers import Input
from kerastuner.engine.hyperparameters import HyperParameters
import tensorflow as tf
'''
class TransformerModel:
    def __init__(self, head_size=128, num_heads=4, ff_dim=2, num_trans_blocks=4, mlp_units=[256], mlp_dropout=0.1, dropout=0.1, attention_axes=1, epsilon=1e-6, kernel_size=1):
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_trans_blocks = num_trans_blocks
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.mlp_dropout = mlp_dropout
        self.attention_axes = attention_axes
        self.epsilon = epsilon
        self.kernel_size = kernel_size
        self.model = self.build_transformer()

    def transformer_encoder(self, inputs):
        x = layers.LayerNormalization(epsilon=self.epsilon)(inputs)
        x = layers.MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout, attention_axes=self.attention_axes)(x, x)
        x = layers.Dropout(self.dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=self.epsilon)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=self.kernel_size, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=self.kernel_size)(x)
        return x + res

    def transformer_decoder(self, inputs, encoded):
        # Masking for inputs
        padding_mask = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.reduce_all(tf.math.equal(x, 0), axis=-1, keepdims=True), tf.float32)
        )(inputs)

        masked_attention_out = layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )(inputs, inputs, attention_mask=padding_mask)

        padding_mask_encoded = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.reduce_all(tf.math.equal(x, 0), axis=-1, keepdims=True), tf.float32)
        )(encoded)

        attention_out = layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )(masked_attention_out, encoded, attention_mask=padding_mask_encoded)

        x = layers.Dropout(self.dropout)(attention_out)
        res = x + masked_attention_out

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=self.epsilon)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=self.kernel_size, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=self.kernel_size)(x)
        return x + res

    def build_transformer(self):
        n_timesteps, n_features, n_outputs = 30, 10, 1
        inputs = tf.keras.Input(shape=(n_timesteps, n_features))
        x = inputs
        for _ in range(self.num_trans_blocks):
            x = self.transformer_encoder(x)

        encoded = x  # Save the encoded representation for later use in the decoder

        for _ in range(self.num_trans_blocks):
            x = self.transformer_decoder(x, encoded)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.mlp_dropout)(x)

        outputs = layers.Dense(n_outputs)(x)
        return tf.keras.Model(inputs, outputs)

    def compile_and_fit(self, X_train, y_train, batch_size=15, epochs=30):
        self.model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["mae"])

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
        start = time.time()
        hist = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks)
        print(time.time() - start)
        return hist
'''
class BayesianTransformerOptimizer:
    def __init__(self, X_train, y_train, max_epochs=30, num_initial_points=5):
        self.X_train = X_train
        self.y_train = y_train
        self.max_epochs = max_epochs
        self.num_initial_points = num_initial_points
        self.best_hps = None

    def build_transformer(self, hp):
        n_timesteps, n_features, n_outputs = 30, 15, 1
        inputs = Input(shape=(n_timesteps, n_features), name='input_layer')
        x = inputs

        for _ in range(hp.Int('num_trans_blocks', min_value=2, max_value=6, step=1)):
            x = self.transformer_encoder(x, hp)
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        mlp_units_choice = hp.Choice('mlp_units', values=[64, 128, 256, 512, 1024])
        # Döngü içinde kullanmak üzere mlp_units_choice değerini alın
        for dim in [mlp_units_choice]:
            x = Dense(dim, activation="relu")(x)
            x = layers.Dropout(hp.Float('mlp_dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
        outputs = Dense(n_outputs, activation="linear")(x)
        model = Model(inputs, outputs)

        model.compile(
            loss="mse",
            optimizer=optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
            metrics=["mae"]
        )
        return model

    def transformer_encoder(self, inputs, hp):
        x = layers.LayerNormalization(epsilon=hp.Float('epsilon', min_value=1e-8, max_value=1e-5, sampling='log'))(inputs)
        x = layers.MultiHeadAttention(
            key_dim=hp.Int('head_size', min_value=64, max_value=256, step=64),
            num_heads=hp.Int('num_heads', min_value=2, max_value=8, step=1),
            dropout=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1),
            attention_axes=hp.Int('attention_axes', min_value=1, max_value=3, step=1)
        )(x, x)
        x = layers.Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
        res =layers.Add()([x, inputs])

        x = layers.LayerNormalization(epsilon=hp.Float('epsilon', min_value=1e-8, max_value=1e-5, sampling='log'))(res)
        x = layers.Conv1D(filters=hp.Int('ff_dim', min_value=1, max_value=4, step=1), kernel_size=hp.Int('kernel_size', min_value=1, max_value=4, step=1), activation="relu")(x)
        x = layers.Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=hp.Int('kernel_size', min_value=1, max_value=4, step=1))(x)
        return x

    def optimize(self):
        tuner = BayesianOptimization(
            self.build_transformer,
            objective='val_loss',
            max_trials=3,
            num_initial_points=self.num_initial_points,
            directory='bayesian_optimization',
            project_name='tranformemodel4'
        )

        tuner.search(self.X_train, self.y_train, epochs=self.max_epochs,
                     validation_split=0.1, batch_size=15, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

        self.best_hps = tuner.oracle.get_best_trials(1)[0].hyperparameters
        
        
class TransformerModel:
    def __init__(self,head_size=128, num_heads=4, ff_dim=2, num_trans_blocks=4, mlp_units=[256], dropout=0.1, mlp_dropout=0.1, attention_axes=1, epsilon=1e-6, kernel_size=1):
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_trans_blocks = num_trans_blocks
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.mlp_dropout = mlp_dropout
        self.attention_axes = attention_axes
        self.epsilon = epsilon
        self.kernel_size = kernel_size
        self.model = self.build_transformer()
        

    def transformer_encoder(self, inputs):
        x = layers.LayerNormalization(epsilon=self.epsilon)(inputs)
        x = layers.MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout, attention_axes=self.attention_axes)(x, x)
        x = layers.Dropout(self.dropout)(x)
        res =layers.Add()([x, inputs])

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=self.epsilon)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=self.kernel_size, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=self.kernel_size)(x)
        return x + res

        

    def build_transformer(self):
        n_timesteps, n_features, n_outputs = 30, 15, 1
        inputs = tf.keras.Input(shape=(n_timesteps, n_features))
        x = inputs
        for _ in range(self.num_trans_blocks):
            x = self.transformer_encoder(x)
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
      
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.mlp_dropout)(x)

        outputs = layers.Dense(n_outputs)(x)
        return tf.keras.Model(inputs, outputs)

    def compile_and_fit(self, X_train, y_train, batch_size=15, epochs=30):
        self.model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003243806953412315),
            metrics=["mae"])

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
        start = time.time()
        hist = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks)
        print(time.time() - start)
        return hist

        










