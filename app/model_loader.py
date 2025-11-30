from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from scipy import sparse
import joblib

from src.model import predict_emotion as _predict_emotion_core
from src.similarity import recommend_songs_from_text as _recommend_core

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

# Project root = one level up from /app
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


# -----------------------------------------------------------------------------
# Resource loading (optional helper for deployment / debugging)
# -----------------------------------------------------------------------------

@lru_cache
def load_resources():
    """
    Load and cache:
      - TF-IDF vectorizer
      - Logistic Regression emotion classifier
      - Song TF-IDF matrix
      - Songs dataframe with predicted emotions

    Note:
        The core modeling logic lives in src.model / src.similarity.
        This helper is mainly useful for debugging or if you want to
        inspect the serving artifacts from a REPL.
    """
    tfidf = joblib.load(MODELS_DIR / "tfidf_emotion.joblib")
    logreg = joblib.load(MODELS_DIR / "logreg_emotion.joblib")
    song_vectors = sparse.load_npz(MODELS_DIR / "song_tfidf_matrix.npz")
    songs_with_pred = pd.read_csv(DATA_CLEAN / "songs_with_predicted_emotions.csv")

    return tfidf, logreg, song_vectors, songs_with_pred


# -----------------------------------------------------------------------------
# Public API used by FastAPI (thin wrappers)
# -----------------------------------------------------------------------------

def predict_emotion(text: str):
    """
    Thin wrapper around src.model.predict_emotion so that the FastAPI app
    only imports from app.model_loader.

    Args:
        text: User input text describing their mood.

    Returns:
        dict with keys:
          - emotion_id
          - emotion
          - confidence
          - (optionally) vector
    """
    return _predict_emotion_core(text)


def recommend_songs_from_text(
    user_text: str,
    top_k: int = 5,
    same_emotion_only: bool = True,
    artist_filter: str | None = None,
    sort_by: str = "similarity",   # "similarity" or "popularity"
):
    """
    Thin wrapper around src.similarity.recommend_songs_from_text.

    This is the main entrypoint used by the /api/recommend endpoint.

    Args:
        user_text: Free-form text describing how the user feels.
        top_k: Number of songs to return.
        same_emotion_only: If True, restrict results to songs whose
                           predicted emotion matches the user emotion.
        artist_filter: Optional artist name to filter recommendations.
        sort_by: Sort key, currently "similarity" or "popularity".

    Returns:
        dict with keys expected by main.py:
          - user_emotion_id
          - user_emotion
          - user_confidence
          - top2_emotions
          - recommendations (list of dicts)
    """
    return _recommend_core(
        user_text=user_text,
        top_k=top_k,
        same_emotion_only=same_emotion_only,
        artist_filter=artist_filter,
        sort_by=sort_by,
    )
