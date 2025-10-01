import pandas as pd
import numpy as np
import matplotlib as matplt
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
# print(f"Number of duplicates: {df.duplicated().sum()}")

df['duration_min'] = df['duration_ms'] / 60000

# print(df)

# Histogram for tempo
# df['tempo'].hist(bins=50)
# plt.title('Tempo Distribution')
# plt.show()

# Boxplot for duration_min
sns.boxplot(y=df['duration_min'])
plt.title('Duration Distribution')
plt.show()

# Genre counts
sns.countplot(y='track_genre', data=df, order=df['track_genre'].value_counts().index[:20])
plt.title('Top 20 Genres')
plt.show()


# *567*123#
