import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras
import statistics

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class LSTM_Model:
    def __init__(self, X_train:np.ndarray, y_train:np.ndarray, X_val:np.ndarray, y_val:np.ndarray, n_labels:int, n_neurons:int = 100, init_mode='he_normal'):
        self.X_train = np.expand_dims(X_train, 1)
        self.y_train = keras.utils.to_categorical(y_train)
        self.X_val = np.expand_dims(X_val, 1)
        self.y_val = keras.utils.to_categorical(y_val)
        self.n_labels = n_labels
        self.n_neurons = n_neurons
        self.init_mode = init_mode
        pass

    def create_model(self):
        self.model = keras.models.Sequential([
            keras.Input(shape=np.shape(self.X_train[0])),
            keras.layers.LSTM(self.n_neurons, activation='relu', kernel_initializer=self.init_mode),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(self.n_labels, activation='softmax')])
        pass

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        self.model.build()
        pass

    def fit(self, verbose:int = 1):
        callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True) 
        self.model.fit(self.X_train, self.y_train, epochs=300, batch_size=128, verbose=verbose, validation_data=(self.X_val, self.y_val), callbacks=callbacks)
        pass

    def predict(self, X_test:np.ndarray):
        self.y_pred = np.argmax(self.model.predict(np.expand_dims(X_test, 1)), axis=1)
        pass

    def eval_model(self, X_test:np.ndarray, y_test:np.ndarray, n_eval_times:int=5):
        f1_set, acc_set = [], []
        for i in range(0, n_eval_times):
            self.model = None
            self.create_model()
            self.compile()
            self.fit(verbose=0)
            self.predict(X_test)
            f1_set.append(sklearn.metrics.f1_score(y_test, self.y_pred, average='macro'))
            acc_set.append(sklearn.metrics.accuracy_score(y_test, self.y_pred))
        f1_mean_std = [statistics.mean(f1_set), statistics.stdev(f1_set)]
        acc_mean_std = [statistics.mean(acc_set), statistics.stdev(acc_set)]
        return (f1_mean_std, acc_mean_std)