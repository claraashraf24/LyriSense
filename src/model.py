# src/model.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import joblib

# Base paths (project root = parent of /src)
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

EMOTION_ID_TO_NAME: Dict[int, str] = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
EMOTION_NAME_TO_ID: Dict[str, int] = {v: k for k, v in EMOTION_ID_TO_NAME.items()}


@lru_cache
def _load_vectorizer_and_model():
    """
    Lazily load the TF-IDF vectorizer and Logistic Regression model
    used for emotion classification.

    This is called once and then cached in memory, which is ideal for
    FastAPI / long-running processes.
    """
    tfidf = joblib.load(MODELS_DIR / "tfidf_emotion.joblib")
    logreg = joblib.load(MODELS_DIR / "logreg_emotion.joblib")
    return tfidf, logreg


def _clean_text(text: str) -> str:
    """
    Minimal text cleaning matching how TF-IDF was trained.

    If you did more preprocessing in the notebook (lowercasing, etc.),
    keep this in sync with that. For now we keep it simple.
    """
    return text.strip()


def predict_emotion(text: str) -> Dict[str, Any]:
    """
    Predict the dominant emotion for a given text.

    This is the main entry point used by:
        - app.model_loader (FastAPI layer)
        - similarity / recommendation code

    Returns a dict with:
        - emotion_id: int            (0–5)
        - emotion: str               ("joy", "sadness", ...)
        - confidence: float          (max class probability)
        - vector: sparse row         (shape (1, n_features)) for similarity
        - top2: List[Tuple[int, str, float]]  (top-2 emotions)

    Raises:
        ValueError: if the input text is empty or not a string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    tfidf, logreg = _load_vectorizer_and_model()

    cleaned = _clean_text(text)

    # One TF-IDF vector for both:
    #   - emotion classification
    #   - similarity search (cosine with songs)
    vec = tfidf.transform([cleaned])  # shape: (1, n_features)

    # Predict probabilities for each emotion class
    probs: np.ndarray = logreg.predict_proba(vec)[0]  # shape: (n_classes,)

    # Dominant emotion
    emotion_id = int(np.argmax(probs))
    emotion_name = EMOTION_ID_TO_NAME.get(emotion_id, "unknown")
    confidence = float(probs[emotion_id])

    # Top-2 emotions (for UI "you might be between X and Y")
    top_two_idx = np.argsort(-probs)[:2]
    top2: List[Tuple[int, str, float]] = []
    for idx in top_two_idx:
        idx_int = int(idx)
        top2.append(
            (
                idx_int,
                EMOTION_ID_TO_NAME.get(idx_int, "unknown"),
                float(probs[idx]),
            )
        )

    return {
        "emotion_id": emotion_id,
        "emotion": emotion_name,
        "confidence": confidence,
        "vector": vec,   # used later by similarity.py
        "top2": top2,
    }
