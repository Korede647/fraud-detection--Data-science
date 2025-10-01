import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data
df = pd.read_csv('Spotify/spotifyData.csv')

# show first 5 rows
# print(df.head())

# data values and find null values 
# df.info()

# data statistics
# print(df.describe())

# missing values
# print(df.isnull().sum())

# Duplicates
print(f"Number of duplicates: {df.duplicated().sum()}")

# *567*123#
