import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras

class LSTM_Classifier:
    def __init__(self, X_train:np.ndarray, y_train:np.ndarray, input_shape:tuple, n_neurons:int = 64):
        self.X_train = X_train
        self.y_train = y_train
        self.input_shape = input_shape
        self.n_neurons = n_neurons
        pass

    def create_model(self):
        model = keras.Sequential()
        model.add(keras.Input(shape=self.input_shape))
        model.add(keras.layers.LSTM(self.n_neurons, activation='tanh',
                                    kernel_initializer=self.init_mode, return_sequences=self.return_sequences,
                                    name="camada_lstm"))

        pass

    def build(self):
        pass