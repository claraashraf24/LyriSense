# src/build_song_embeddings.py
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

# ---------- CONFIG ----------
# This file lives in: <project_root>/src/build_song_embeddings.py
# So project root is one level up from this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_CLEAN = PROJECT_ROOT / "data" / "cleaned"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_CLEAN / "songs_with_predicted_emotions.csv"
OUT_PATH = MODELS_DIR / "song_embeddings_openai.npy"

MODEL_NAME = "text-embedding-3-small"  # cheap, good
MAX_LYRICS_CHARS = 4000                # truncate long lyrics
# ----------------------------

client = OpenAI()  # uses OPENAI_API_KEY from env


def get_text(row) -> str:
    # prefer clean_lyrics, fallback to lyrics
    text = str(row.get("clean_lyrics") or row.get("lyrics") or "").strip()
    if not text:
        title = str(row.get("title") or "")
        artist = str(row.get("artist") or "")
        text = f"{title} {artist}"
    # Truncate for safety
    return text[:MAX_LYRICS_CHARS]


def main():
    print(f"Loading songs from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Total songs: {len(df)}")

    embeddings = []
    for idx, row in df.iterrows():
        text = get_text(row)
        if not text.strip():
            # if totally empty, just use zeros for now
            if idx == 0:
                # we don't know dim yet, make a dummy call
                resp = client.embeddings.create(
                    model=MODEL_NAME,
                    input="dummy",
                )
                dim = len(resp.data[0].embedding)
            else:
                dim = len(embeddings[0])
            vec = np.zeros(dim, dtype="float32")
            embeddings.append(vec)
            continue

        print(f"[{idx+1}/{len(df)}] embedding: {row.get('title', 'Unknown')}")
        resp = client.embeddings.create(
            model=MODEL_NAME,
            input=text,
        )
        vec = np.array(resp.data[0].embedding, dtype="float32")
        embeddings.append(vec)

    emb_matrix = np.vstack(embeddings)
    print(f"Embeddings shape: {emb_matrix.shape}")
    np.save(OUT_PATH, emb_matrix)
    print(f"Saved embeddings to: {OUT_PATH}")


if __name__ == "__main__":
    main()
