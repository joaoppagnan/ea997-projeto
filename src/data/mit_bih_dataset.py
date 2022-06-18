import numpy as np
import pandas as pd
import os

class MIT_BIH_Dataset:
    def __init__(self, path:str, window_size:int=180):
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
        for r in range(0,len(self.records)):
            signals = []

            with open(self.records[r], 'rt') as csvfile:
                spamreader = pd.read_csv(csvfile, delimiter=',', quotechar='|') # read CSV file\
                row_index = -1
                for row in spamreader:
                    if(row_index >= 0):
                        signals.insert(row_index, int(row[1]))
                    row_index += 1

            # Read anotations: R position and Arrhythmia class
            example_beat_printed = False
            with open(self.annotations[r], 'r') as fileID:
                data = fileID.readlines() 
                beat = list()

                for d in range(1, len(data)): # 0 index is Chart Head
                    splitted = data[d].split(' ')
                    splitted = filter(None, splitted)
                    next(splitted) # Time... Clipping
                    pos = int(next(splitted)) # Sample ID
                    arrhythmia_type = next(splitted) # Type
                    if(arrhythmia_type in self.classes):
                        arrhythmia_index = self.classes.index(arrhythmia_type)
                        self.count_classes[arrhythmia_index] += 1
                        if(self.window_size <= pos and pos < (len(signals) - self.window_size)):
                            beat = signals[pos-self.window_size:pos+self.window_size]     ## REPLACE WITH R-PEAK DETECTION
                            self.X_data.append(beat)
                            self.y_data.append(arrhythmia_index)
        pass

    def get_filenames_annotations(self):
        return (self.filenames, self.annotations)

    def get_X_data(self):
        return self.X_data

    def get_y_data(self):
        return self.y_data