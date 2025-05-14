import pandas as pd


df = pd.read_csv('universal_top_spotify_songs.csv')
print(df.head())
print(df.info())


df = df.dropna(subset=['popularity', 'danceability', 'valence', 'energy', 'tempo'])
if 'year' in df.columns:
    df = df[df['year'].between(2010, 2024)]
df = df.reset_index(drop=True)
print("\nCleaned dataset shape:", df.shape)
print("Columns after cleaning:", df.columns.tolist())



from datetime import datetime
df['album_release_date'] = pd.to_datetime(df['album_release_date'], errors='coerce')
df['release_year'] = df['album_release_date'].dt.year
current_year = datetime.now().year
df['song_age'] = current_year - df['release_year']
df['average_popularity_per_artist'] = df.groupby('artists')['popularity'].transform('mean')
df['lyric_word_count'] = None
df['danceability_to_lyric_ratio'] = None
print(df[['name', 'artists', 'release_year', 'song_age', 'average_popularity_per_artist']].head())





df['artists_split'] = df['artists'].str.split(', ')
df_exploded = df.explode('artists_split')
df_exploded['average_popularity_per_artist_clean'] = df_exploded.groupby('artists_split')['popularity'].transform('mean')
print(df_exploded[['name', 'artists_split', 'average_popularity_per_artist_clean']].head())





#Main Sprint 2

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --- Pearson correlation ---
danceability = df['danceability']
popularity = df['popularity']

correlation, p_value = pearsonr(danceability, popularity)
print(f"Pearson correlation between danceability and popularity: {correlation:.2f}")
print(f"P-value: {p_value:.4f}")

# --- Visualization 1: Popularity Histogram ---
plt.figure(figsize=(8, 5))
sns.histplot(popularity, bins=50, kde=True)
plt.title('Distribution of Popularity Scores')
plt.xlabel('Popularity')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.savefig("popularity_histogram.png")
plt.close()

# --- Visualization 2: Danceability vs Popularity Scatter Plot ---
plt.figure(figsize=(8, 5))
sns.scatterplot(x=danceability, y=popularity, alpha=0.3)
plt.title('Danceability vs. Popularity')
plt.xlabel('Danceability')
plt.ylabel('Popularity')
plt.grid(True)
plt.tight_layout()
plt.savefig("danceability_vs_popularity.png")
plt.close()

# --- Visualization 3: Correlation Heatmap (for numeric features) ---
plt.figure(figsize=(10, 8))
correlation_matrix = df[['popularity', 'danceability', 'energy', 'valence', 'tempo']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
