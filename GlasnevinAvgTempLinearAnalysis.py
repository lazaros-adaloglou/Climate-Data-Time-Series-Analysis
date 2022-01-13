# Imports.
import LinearAnalysisFunctions as lf
import pandas as pd
import numpy as np
import os

# Get Path.
print("---------------------------------------------------------------------------------------------------------------")
print('Current Path', os.getcwd())

# Path to Time Series.
filename = 'data/Glasnevin_Data.csv'

# Preprocess Average Temperature Time Series.
try:
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
except FileNotFoundError:
    os.system('python GlasnevinDataPreprocessing.py')
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
print("---------------------------------------------------------------------------------------------------------------")
print("Time Series:")
data = data.drop("Unnamed: 0", axis=1)
print(data)

xV = data.AvgTemp
