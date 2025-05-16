# Functions to load and split the data
import pandas as pd
from sklearn.model_selection import train_test_split
from sem_proj.config import RAW_DATA_PATH

def load_data():
    df = pd.read_csv(RAW_DATA_PATH)

    
    df = df.dropna(subset=[
        "danceability", "energy", "valence", "tempo", "popularity"
    ])

    
    features = ["danceability", "energy", "valence", "tempo"]
    target = "popularity"

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)
