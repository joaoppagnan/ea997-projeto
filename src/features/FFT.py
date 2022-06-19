import numpy as np

class FFT():
    def __init__(self, X_data:np.ndarray):
        self.X_data = X_data

    def transform(self):
            self.X_fft = [np.abs(np.fft.fft(x))[:int(len(x)/2)] for x in self.X_data]

    def get_transformed_data(self):
        return self.X_fft