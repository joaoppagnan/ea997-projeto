import numpy as np
import pandas as pd
import os
import pywt

class MIT_BIH_Dataset:
    def __init__(self, path:str, window_size:int=180, maximum_counting:int=10000):
        self.path = path
        self.window_size = window_size
        self.maximum_counting = maximum_counting
        self.classes = ['N', 'L', 'R', 'A', 'V']
        self.n_classes = len(self.classes)
        self.count_classes = [0]*self.n_classes
        self.X = list()
        self.y = list()

    def denoise(self): 
        w = pywt.Wavelet('sym4')
        maxlev = pywt.dwt_max_level(len(self.X), w.dec_len)
        threshold = 0.04 # Threshold for filtering

        coeffs = pywt.wavedec(self.X, 'sym4', level=maxlev)
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
            
        datarec = pywt.waverec(coeffs, 'sym4')
        
        return datarec

    def get_filenames_annotations(self):
        # Read files
        filenames = next(os.walk(self.path))[2]

        # Split and save .csv , .txt 
        self.records = list()
        self.annotations = list()
        filenames.sort()

        # segrefating filenames and annotations
        for f in filenames:
            filename, file_extension = os.path.splitext(f)
            # *.csv
            if(file_extension == '.csv'):
                self.records.append(self.path + filename + file_extension)
            # *.txt
            else:
                self.annotations.append(self.path + filename + file_extension)
