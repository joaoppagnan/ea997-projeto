import numpy as np
import pandas as pd
import os

class MIT_BIH_Dataset:
    def __init__(self, path:str):
        self.path = path
        self.classes = ['N', 'L', 'R', 'A', 'V']
        self.n_classes = len(self.classes)
        self.count_classes = [0]*self.n_classes
        self.X_data = list()
        self.y_data = list()
        self.process_filenames_annotations()
        self.extract_data()
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

    def extract_data(self):
        pass

    def get_filenames_annotations(self):
        return (self.filenames, self.annotations)