# Imports.
import os
import pandas as pd
import numpy as np

filename = 'data/Glasnevin_Data.csv'

try:
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
except FileNotFoundError:
    os.system('python GlasnevinDataPreprocessing.py')
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])

data = data.drop("Unnamed: 0", axis=1)
print(data)

