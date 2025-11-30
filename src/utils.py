# src/utils.py
from __future__ import annotations

"""
utils.py

Utility module for loading shared data artifacts used across LyriSense:
    - songs_with_pred: DataFrame of songs with predicted emotions
    - song_vectors: sparse TF-IDF matrix for song lyrics

These are loaded from:
    data/cleaned/songs_with_predicted_emotions.csv
    models/song_tfidf_matrix.npz

The paths are resolved relative to the project root, so the code works
on any machine (no hard-coded D: paths).
"""

from pathlib import Path

import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# Paths (project root = parent of /src)
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_CLEAN = BASE_DIR / "data" / "cleaned"
MODELS_DIR = BASE_DIR / "models"

SONGS_WITH_PRED_PATH = DATA_CLEAN / "songs_with_predicted_emotions.csv"
SONG_VECTORS_PATH = MODELS_DIR / "song_tfidf_matrix.npz"


# ---------------------------------------------------------------------------
# Load artifacts once at import time
# ---------------------------------------------------------------------------

# DataFrame with song metadata + predicted emotions from modeling notebook
songs_with_pred = pd.read_csv(SONGS_WITH_PRED_PATH)

# Sparse TF-IDF matrix for song lyrics (same order as songs_with_pred)
song_vectors = sparse.load_npz(SONG_VECTORS_PATH)
