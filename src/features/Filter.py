import numpy as np
import pywt

class Filter():
    def __init__(self, data:np.ndarray):
        self.data = data
    
    def denoise_data(self): 
        w = pywt.Wavelet('sym4')
        maxlev = pywt.dwt_max_level(len(self.data), w.dec_len)
        threshold = 0.04 # Threshold for filtering

        coeffs = pywt.wavedec(self.data, 'sym4', level=maxlev)
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold*coeffs[i].max())
            
        self.clean_data = pywt.waverec(coeffs, 'sym4')
        pass

    def get_clean_data(self):
        return self.clean_data