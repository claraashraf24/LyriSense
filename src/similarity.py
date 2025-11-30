import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.model import predict_emotion
from src.utils import (
    songs_with_pred,
    song_vectors,
)
from src.spotify_helpers import fetch_spotify_metadata


EMOTION_KEYWORDS = {
    "sadness": [
        "sad", "cry", "crying", "alone", "lonely", "hurt",
        "tears", "broken", "goodbye", "lost", "empty", "missing you"
    ],
    "joy": [
        "happy", "happiness", "joy", "smile", "laugh", "sunshine",
        "love my life", "feels so good", "celebrate"
    ],
    "love": [
        "love", "loving you", "heartbeat", "kiss", "together",
        "baby", "darling", "hold you", "forever"
    ],
    "anger": [
        "angry", "rage", "hate", "screaming", "fighting", "fight",
        "mad at", "pissed", "burn", "destroy"
    ],
    "fear": [
        "afraid", "scared", "fear", "anxious", "anxiety", "worry",
        "nightmare", "darkness", "run away", "danger"
    ],
    "surprise": [
        "suddenly", "out of nowhere", "didn't expect", "surprise",
        "shocked", "again and again", "can't believe"
    ],
}


def extract_lyric_snippet(row, max_len: int = 200) -> str | None:
    """
    Try to extract a short snippet of lyrics that supports the predicted emotion.
    """
    # Prefer raw 'lyrics' if present, otherwise 'clean_lyrics'
    text = str(row.get("lyrics") or row.get("clean_lyrics") or "").strip()
    if not text:
        return None

    emotion = str(row.get("pred_emotion") or "").lower()
    keywords = EMOTION_KEYWORDS.get(emotion, [])

    lower_text = text.lower()

    # 1) Try to find a keyword window
    for kw in keywords:
        idx = lower_text.find(kw)
        if idx != -1:
            start = max(0, idx - 60)
            end = min(len(text), idx + 80)
            snippet = text[start:end].replace("\n", " ").strip()
            if len(snippet) > max_len:
                snippet = snippet[:max_len] + "..."
            return snippet

    # 2) Fallback: first non-empty lines or first max_len chars
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        snippet = " ".join(lines[:2])
    else:
        snippet = text[:max_len]

    snippet = snippet.replace("\n", " ").strip()
    if len(snippet) > max_len:
        snippet = snippet[:max_len] + "..."
    return snippet or None



def recommend_songs_from_text(
    user_text: str,
    top_k: int = 5,
    same_emotion_only: bool = True,
    min_same_emotion: int = 10,
    artist_filter: str | None = None,
    sort_by: str = "similarity",   # "similarity" or "popularity"
):
    """
    Main recommendation engine:
      1) Predict emotion from text
      2) Filter songs by emotion (fallback to all songs)
      3) Optional: filter by artist / country
      4) Compute cosine similarities
      5) Sort by similarity or popularity
      6) Return top_k songs, each with a lyric snippet.
    """

    # --- 1. Predict emotion ---
    pred = predict_emotion(user_text)
    user_vec = pred["vector"]
    user_emotion_id = pred["emotion_id"]
    top2 = pred["top2"]   # list of (id, name, conf)

    # --- 2. Filter songs by emotion (optional) ---
    if same_emotion_only:
        mask = songs_with_pred["pred_emotion_id"] == user_emotion_id
        idx = np.where(mask.values)[0]

        if len(idx) < min_same_emotion:
            filtered_vectors = song_vectors
            filtered_songs = songs_with_pred.reset_index(drop=True)
        else:
            filtered_vectors = song_vectors[idx]
            filtered_songs = songs_with_pred.iloc[idx].reset_index(drop=True)
    else:
        filtered_vectors = song_vectors
        filtered_songs = songs_with_pred.reset_index(drop=True)

    # --- 2b. Artist filter (optional) ---
    if artist_filter:
        artist_lower = artist_filter.lower()
        mask_artist = filtered_songs["artist"].str.lower().str.contains(
            artist_lower, na=False
        )
        idx_artist = np.where(mask_artist.values)[0]
        filtered_vectors = filtered_vectors[idx_artist]
        filtered_songs = filtered_songs.iloc[idx_artist].reset_index(drop=True)


    # If after filtering there is nothing left, just return empty recs
    if len(filtered_songs) == 0:
        return {
            "user_emotion_id": user_emotion_id,
            "user_emotion": pred["emotion"],
            "user_confidence": pred["confidence"],
            "top2_emotions": top2,
            "recommendations": [],
        }

    # --- 3. Similarity ---
    sims = cosine_similarity(user_vec, filtered_vectors)[0]  # shape (n_songs,)

    # Attach similarity to the DataFrame
    filtered_songs = filtered_songs.copy()
    filtered_songs["similarity"] = sims

    # --- 4. Sorting logic ---
    sort_by = (sort_by or "similarity").lower()

    if sort_by == "popularity" and "popularity" in filtered_songs.columns:
        # Sort by popularity (desc), then similarity (desc)
        sorted_songs = filtered_songs.sort_values(
            by=["popularity", "similarity"],
            ascending=[False, False],
        )
    else:
        # Sort by similarity (desc); if popularity exists, use it as tie-breaker
        if "popularity" in filtered_songs.columns:
            sorted_songs = filtered_songs.sort_values(
                by=["similarity", "popularity"],
                ascending=[False, False],
            )
        else:
            sorted_songs = filtered_songs.sort_values(
                by="similarity",
                ascending=False,
            )

        top_songs = sorted_songs.head(top_k)

    # --- 5. Build recommendations ---
    recs = []
    for _, row in top_songs.iterrows():
        snippet = extract_lyric_snippet(row, max_len=200)

        recs.append({
            "title": row["title"],
            "artist": row["artist"],
            "album": row.get("album"),
            "song_emotion": row.get("pred_emotion"),
            "song_emotion_conf": float(row.get("pred_emotion_conf", 0.0)),
            "similarity": float(row["similarity"]),
            "lyric_snippet": snippet or "",
            "popularity": float(row.get("popularity", 0.0)) if "popularity" in row else None,
        })

    return {
        "user_emotion_id": user_emotion_id,
        "user_emotion": pred["emotion"],
        "user_confidence": pred["confidence"],
        "top2_emotions": top2,
        "recommendations": recs,
    }
