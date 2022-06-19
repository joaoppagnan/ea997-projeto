import numpy as np
import csv
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import os

class MIT_BIH_Dataset:
    def __init__(self, path:str, window_size:int=180, normalize:bool=False):
        self.path = path
        self.window_size = window_size
        self.classes = ['N', 'L', 'R', 'A', 'V']
        self.n_classes = len(self.classes)
        self.count_classes = [0]*self.n_classes
        self.X_data = list()
        self.y_data = list()
        self.normalize = normalize
        self.process_filenames_annotations()
        self.extract_data()
        self.rebalance_data()
        pass

    def process_filenames_annotations(self):
        # Read files
        self.filenames = next(os.walk(self.path))[2]

        # Split and save .csv , .txt 
        self.records = list()
        self.annotations = list()
        self.filenames.sort()

        # segrefating self.filenames and annotations
        for f in self.filenames:
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
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                row_index = -1
                for row in spamreader:
                    if(row_index >= 0):
                        signals.insert(row_index, int(row[1]))
                    row_index += 1

            if self.normalize:
                signals = stats.zscore(signals)
            signals = np.array(signals)

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

    def rebalance_data(self):
        df = pd.DataFrame({'MLII':self.X_data, 'Class':self.y_data})
        df_balanced = []
        histogram = df.groupby(df["Class"], as_index=False).size()
        random_state = 0
        for c in histogram["Class"]:
            temp = df.loc[df["Class"] == c]
            temp = temp.sample(n=histogram["size"].min(), random_state=random_state)
            df_balanced.append(temp)
            random_state += 1
        df = pd.DataFrame(np.vstack(df_balanced))
        df.columns = ['MLII','Class']
        X_data = np.array(df['MLII'])
        self.X_data = [x.flatten() for x in X_data]
        self.y_data = np.array(df['Class']).astype('int')
        pass

    def get_filenames_annotations(self):
        return (self.filenames, self.annotations)

    def get_X_data(self):
        return self.X_data

    def get_y_data(self):
        return self.y_data

    def get_df(self):
        return self.df