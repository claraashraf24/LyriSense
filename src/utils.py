from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse
import joblib

BASE_DIR = Path(r"D:\Sheridan\First semester\Python\LyriSense")

DATA_CLEAN = BASE_DIR / "data" / "cleaned"
MODELS_DIR = BASE_DIR / "models"

# Load datasets
songs_with_pred = pd.read_csv(DATA_CLEAN / "songs_with_predicted_emotions.csv")

# Load TF-IDF matrix for songs
song_vectors = sparse.load_npz(MODELS_DIR / "song_tfidf_matrix.npz")
