import pandas as pd
import numpy as np
import matplotlib as matplt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, confusion_matrix  # For pseudo-evaluation
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

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
# sns.boxplot(y=df['duration_min'])
# plt.title('Duration Distribution')
# plt.show()

# Genre counts
# sns.countplot(y='track_genre', data=df, order=df['track_genre'].value_counts().index[:20])
# plt.title('Top 20 Genres')
# plt.show()

# Scatter plot: energy vs danceability
# sns.scatterplot(x='danceability', y='energy', data=df, hue='track_genre', alpha=0.5)
# plt.title('Energy vs Danceability by Genre')
# plt.show()

# Boxplot for loudness by explicit
# sns.boxplot(x='explicit', y='loudness', data=df)
# plt.title('Loudness by Explicit Content')
# plt.show()

numeric_cols = ['popularity', 'duration_min', 'danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
# corr = df[numeric_cols].corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()

# IQR for tempo
Q1 = df['tempo'].quantile(0.25)
Q3 = df['tempo'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['tempo'] < (Q1 - 1.5 * IQR)) | (df['tempo'] > (Q3 + 1.5 * IQR))]
# print(f"Outliers in tempo: {len(outliers)}")

# Visualize outliers in boxplots for all numerics
# df[numeric_cols].plot(kind='box', subplots=True, layout=(3,4), figsize=(12,8))
# plt.show()
df[numeric_cols].plot(kind='box', subplots=True, layout=(3,4), figsize=(12,8))
plt.suptitle('Boxplots of Numeric Features')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent title overlap
# plt.show()



# New features
# df['duration_min'] = df['duration_ms'] / 60000  # Done in cleaning
df['log_loudness'] = np.log1p(-df['loudness'] + 60)  # Handle negative dB, skew
df['energy_acoustic_ratio'] = df['energy'] / (df['acousticness'] + 1e-8)  # Intensity vs acoustic
df['speech_instrumental'] = df['speechiness'] + df['instrumentalness']  # Combined non-vocal metric

# Encode categoricals
df['explicit'] = df['explicit'].astype(int)
le = LabelEncoder()
df['track_genre'] = le.fit_transform(df['track_genre'])
df['mode'] = LabelEncoder().fit_transform(df['mode'])  # If not already numeric
df['key'] = LabelEncoder().fit_transform(df['key'])  # Treat as categorical

# Drop originals
df.drop(['duration_ms'], axis=1, inplace=True)

# Scale (important for anomaly detection)
scaler = StandardScaler()
feature_cols = ['popularity', 'duration_min', 'danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                'log_loudness', 'energy_acoustic_ratio', 'speech_instrumental']
df[feature_cols] = scaler.fit_transform(df[feature_cols])


# Features (exclude genre if not using as feature)
X = df[feature_cols]

# Train Isolation Forest (assume 1% anomalies)
model = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = model.fit_predict(X)  # -1 for anomaly, 1 for normal

# "Evaluation" - since unsupervised, visualize
# PCA for 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['anomaly'], palette={1:'blue', -1:'red'})
plt.title('PCA Projection with Anomalies')
plt.show()

# Silhouette score (cluster quality, treating anomalies as cluster)
print(f"Silhouette Score: {silhouette_score(X, df['anomaly']):.4f}")

# "Metrics" - proportion flagged
print(f"Anomalies detected: { (df['anomaly'] == -1).sum() / len(df):.2%}")

# Pseudo-confusion if simulating labels (e.g., assume high instrumentalness as anomaly)
sim_anomaly = (df['instrumentalness'] > 2)  # Z-score >2 as proxy
cm = confusion_matrix(sim_anomaly, df['anomaly'] == -1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Pseudo-Confusion Matrix')
plt.show()

# *567*123#
