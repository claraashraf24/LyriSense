# src/model.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import joblib

# NOTE: no more OpenAI import here.
# from .openai_helpers import rewrite_feeling


# Base paths (project root = parent of /src)
BASE_DIR = Path(__file__).resolve().parents[1]
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


@lru_cache
def _load_vectorizer_and_model():
    """
    Lazily load the TF-IDF vectorizer and Logistic Regression model
    used for emotion classification.
    """
    tfidf = joblib.load(MODELS_DIR / "tfidf_emotion.joblib")
    logreg = joblib.load(MODELS_DIR / "logreg_emotion.joblib")
    return tfidf, logreg


def _clean_text(text: str) -> str:
    # Keep it very simple; match how you trained TF-IDF in the notebook.
    return text.strip()


def predict_emotion(text: str) -> Dict[str, Any]:
    """
    Main emotion prediction function used by both the API and similarity engine.

    Returns a dict with:
      - emotion_id: int
      - emotion: str
      - confidence: float
      - vector: np.ndarray shape (1, n_features)  # used for similarity
      - top2: List[Tuple[int, str, float]]
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    tfidf, logreg = _load_vectorizer_and_model()

    # ✅ SIMPLE VERSION: use the original user text only
    cleaned = _clean_text(text)

    # One TF-IDF vector for both:
    #  - emotion classification
    #  - similarity search
    vec = tfidf.transform([cleaned])  # shape: (1, n_features)

    # Predict emotion using your sklearn model
    probs = logreg.predict_proba(vec)[0]  # shape: (n_classes,)

    emotion_id = int(np.argmax(probs))
    emotion_name = EMOTION_ID_TO_NAME.get(emotion_id, "unknown")
    confidence = float(probs[emotion_id])

    # Top-2 emotions
    top_two_idx = np.argsort(-probs)[:2]
    top2: List[tuple[int, str, float]] = []
    for idx in top_two_idx:
        top2.append(
            (
                int(idx),
                EMOTION_ID_TO_NAME.get(int(idx), "unknown"),
                float(probs[idx]),
            )
        )

    return {
        "emotion_id": emotion_id,
        "emotion": emotion_name,
        "confidence": confidence,
        "vector": vec,   # ✅ this is what similarity.py uses
        "top2": top2,
    }
