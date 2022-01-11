# Imports.
import os
import pandas as pd
import numpy as np

filename = 'data/DublinAirport_Data.csv'

try:
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])
except FileNotFoundError:
    os.system('python DublinAirportDataPreprocessing.py')
    data = pd.read_csv(filename, delimiter=',', parse_dates=['date'])

print("---------------------------------------------------------------------------------------------------------------")
print("Time Series:")
data = data.drop("Unnamed: 0", axis=1)
print(data)

