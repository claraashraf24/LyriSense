"""
build_song_embeddings.py

Optional utility script to generate dense song embeddings using OpenAI's
text-embedding-3-small model.

These embeddings are NOT required for the main LyriSense pipeline used in
the final project:
    - TF-IDF + Logistic Regression for emotion classification
    - TF-IDF + cosine similarity for recommendations

This script is just for experimentation / future extensions.
It expects:
    - OPENAI_API_KEY set in the environment
    - data/cleaned/songs_with_predicted_emotions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI


# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

# This file lives in: <project_root>/src/build_song_embeddings.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_CLEAN = PROJECT_ROOT / "data" / "cleaned"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_CLEAN / "songs_with_predicted_emotions.csv"
OUT_PATH = MODELS_DIR / "song_embeddings_openai.npy"

MODEL_NAME = "text-embedding-3-small"  # lightweight, good quality
MAX_LYRICS_CHARS = 4000                # truncate long lyrics to fit limits


# OpenAI client (uses OPENAI_API_KEY from env)
client = OpenAI()


def get_text(row: pd.Series) -> str:
    """
    Build the text to embed for a given song row.

    Preference order:
        1) clean_lyrics
        2) lyrics
        3) "<title> <artist>" fallback

    Lyrics are truncated to MAX_LYRICS_CHARS for safety.
    """
    text = str(row.get("clean_lyrics") or row.get("lyrics") or "").strip()

    if not text:
        title = str(row.get("title") or "")
        artist = str(row.get("artist") or "")
        text = f"{title} {artist}".strip()

    return text[:MAX_LYRICS_CHARS]


def main() -> None:
    print(f"Loading songs from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Total songs: {len(df)}")

    embeddings: list[np.ndarray] = []
    dim: int | None = None

    for idx, row in df.iterrows():
        text = get_text(row)

        # If we somehow have completely empty text, fill with zeros
        if not text.strip():
            if dim is None:
                # Probe once to determine embedding dimension
                resp = client.embeddings.create(
                    model=MODEL_NAME,
                    input="dummy",
                )
                dim = len(resp.data[0].embedding)

            vec = np.zeros(dim, dtype="float32")
            embeddings.append(vec)
            continue

        print(f"[{idx + 1}/{len(df)}] embedding: {row.get('title', 'Unknown')}")

        resp = client.embeddings.create(
            model=MODEL_NAME,
            input=text,
        )
        emb = np.array(resp.data[0].embedding, dtype="float32")

        # Initialize dim if this is the first successful call
        if dim is None:
            dim = emb.shape[0]

        embeddings.append(emb)

    emb_matrix = np.vstack(embeddings)
    print(f"Embeddings shape: {emb_matrix.shape}")
    np.save(OUT_PATH, emb_matrix)
    print(f"Saved embeddings to: {OUT_PATH}")


if __name__ == "__main__":
    main()
