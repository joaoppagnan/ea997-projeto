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
        self.original_data = list()
        self.y = list()
        pass

    def denoise_data(self): 
        w = pywt.Wavelet('sym4')
        maxlev = pywt.dwt_max_level(len(self.original_data), w.dec_len)
        threshold = 0.04 # Threshold for filtering

        coeffs = pywt.wavedec(self.original_data, 'sym4', level=maxlev)
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
            
        self.clean_data = pywt.waverec(coeffs, 'sym4')
        pass

    def process_filenames_annotations(self):
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
        pass

    def get_clean_data(self):
        return self.clean_data

    def get_filenames_annotations(self):
        return (self.filenames, self.annotations)