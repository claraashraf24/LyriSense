from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy import sparse
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from src.model import predict_emotion as _predict_emotion_core
from src.similarity import recommend_songs_from_text as _recommend_core


# Base paths (project root = one level up from /app)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_CLEAN = BASE_DIR / "data" / "cleaned"
MODELS_DIR = BASE_DIR / "models"

EMOTION_ID_TO_NAME = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
EMOTION_NAME_TO_ID = {v: k for k, v in EMOTION_ID_TO_NAME.items()}

# At the top
EMOJI_TO_EMOTION_ID = {
    "😊": 1, "😄": 1, "😁": 1, "😃": 1, "🥰": 2, "😍": 2,
    "😭": 0, "😢": 0, "😞": 0, "😔": 0,
    "😡": 3, "🤬": 3,
    "😱": 4, "😨": 4, "😰": 4, "😟": 4,
    "😮": 5, "😲": 5, "😯": 5, "🤯": 5,
}


@lru_cache
def load_resources():
    """
    Load and cache:
    - TF-IDF vectorizer
    - Logistic Regression emotion classifier
    - Song TF-IDF matrix
    - Songs dataframe with predicted emotions
    """
    tfidf = joblib.load(MODELS_DIR / "tfidf_emotion.joblib")
    logreg = joblib.load(MODELS_DIR / "logreg_emotion.joblib")
    song_vectors = sparse.load_npz(MODELS_DIR / "song_tfidf_matrix.npz")
    songs_with_pred = pd.read_csv(DATA_CLEAN / "songs_with_predicted_emotions.csv")

    return tfidf, logreg, song_vectors, songs_with_pred


def predict_emotion(text: str):
    return _predict_emotion_core(text)



def _extract_emoji_emotions(text: str):
    emoji_ids = []
    chars = []

    for ch in text:
        if ch in EMOJI_TO_EMOTION_ID:
            emoji_ids.append(EMOJI_TO_EMOTION_ID[ch])
        else:
            chars.append(ch)

    cleaned_text = "".join(chars)
    return cleaned_text, emoji_ids



def recommend_songs_from_text(
    user_text: str,
    top_k: int = 5,
    same_emotion_only: bool = True,
    artist_filter: str | None = None,
    sort_by: str = "similarity",   # "similarity" or "popularity"
):
    """
    Thin wrapper around src.similarity.recommend_songs_from_text
    so the FastAPI app only imports from app.model_loader.
    """
    return _recommend_core(
        user_text=user_text,
        top_k=top_k,
        same_emotion_only=same_emotion_only,
        artist_filter=artist_filter,
        sort_by=sort_by,
    )

