import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load the toy box
df = pd.read_csv('Spotify/spotifyData.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(f"Number of duplicates: {df.duplicated().sum()}")

# Step 2: Clean the toy box
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['duration_min'] = df['duration_ms'] / 60000
df.drop(['track_id', 'track_name', 'album_name', 'artists', 'duration_ms'], axis=1, inplace=True)

# Step 3: Play with toys (EDA)
df['tempo'].hist(bins=50)
plt.title('How Fast Songs Are')
plt.show()
sns.boxplot(y=df['duration_min'])
plt.title('How Long Songs Are')
plt.show()
sns.countplot(y='track_genre', data=df, order=df['track_genre'].value_counts().index[:20])
plt.title('Top 20 Music Types')
plt.show()
sns.scatterplot(x='danceability', y='energy', data=df, hue='track_genre', alpha=0.5)
plt.title('Dance vs Energy')
plt.show()
sns.boxplot(x='explicit', y='loudness', data=df)
plt.title('How Loud Explicit Songs Are')
plt.show()
numeric_cols = ['popularity', 'duration_min', 'danceability', 'energy', 'loudness', 
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('How Toys Connect')
plt.show()
Q1 = df['tempo'].quantile(0.25)
Q3 = df['tempo'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['tempo'] < (Q1 - 1.5 * IQR)) | (df['tempo'] > (Q3 + 1.5 * IQR))]
print(f"Weird tempo songs: {len(outliers)}")
df[numeric_cols].plot(kind='box', subplots=True, layout=(3,4), figsize=(12,8))
plt.suptitle('Boxplots of Song Toys')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Step 4: Make new toys and anomaly label
df['anomaly'] = (df['instrumentalness'] > df['instrumentalness'].quantile(0.99)).astype(int)
print(f"Weird songs: {df['anomaly'].sum()}")
df['log_loudness'] = np.log1p(-df['loudness'] + 60)
df['energy_acoustic_ratio'] = df['energy'] / (df['acousticness'] + 1e-8)
df['speech_instrumental'] = df['speechiness'] + df['instrumentalness']
df['explicit'] = df['explicit'].astype(int)
le = LabelEncoder()
df['track_genre'] = le.fit_transform(df['track_genre'])
df['mode'] = LabelEncoder().fit_transform(df['mode'])
df['key'] = LabelEncoder().fit_transform(df['key'])
scaler = StandardScaler()
feature_cols = ['popularity', 'duration_min', 'danceability', 'energy', 'loudness', 
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 
                'tempo', 'log_loudness', 'energy_acoustic_ratio', 'speech_instrumental']
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Step 5: Teach the robot
X = df[feature_cols]
y = df['anomaly']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall: {recall_score(y_test, y_pred):.2%}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2%}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.2%}")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Mistakes the Robot Made')
plt.xlabel('Guessed (0=Normal, 1=Weird)')
plt.ylabel('Real (0=Normal, 1=Weird)')
plt.show()

# Step 6: What the robot learned
importances = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})
importances = importances.sort_values('Importance', ascending=False)
print(importances.head(10))
sns.barplot(x='Importance', y='Feature', data=importances.head(10))
plt.title('Best Toys for Finding Weird Songs')
plt.show()