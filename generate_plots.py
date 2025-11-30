"""
generate_plots.py

Utility script to generate all plots for the LyriSense project and save them
to results/plots/.

Run from the project root:

    python generate_plots.py
    This is a supporting script
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from scipy import sparse


# -----------------------------------------------------------------------------
# Paths & setup
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_CLEAN = BASE_DIR / "data" / "cleaned"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_ID_TO_NAME = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


def save_fig(name: str):
    """Helper to save the current matplotlib figure in results/plots/."""
    out_path = RESULTS_DIR / name
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# 1. Emotion training distribution (from emotion_clean.csv)
# -----------------------------------------------------------------------------

def plot_emotion_training_distribution():
    path = DATA_CLEAN / "emotion_clean.csv"
    df = pd.read_csv(path)

    if "emotion_id" not in df.columns:
        raise ValueError(f"'emotion_id' column not found in {path}")

    counts = df["emotion_id"].value_counts().sort_index()
    labels = [EMOTION_ID_TO_NAME.get(i, str(i)) for i in counts.index]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=counts.values)
    plt.xlabel("Emotion")
    plt.ylabel("Number of samples")
    plt.title("Emotion Training Distribution (emotion_clean.csv)")
    save_fig("emotion_training_distribution.png")


# -----------------------------------------------------------------------------
# 2. Song predicted emotion distribution (from songs_with_predicted_emotions.csv)
# -----------------------------------------------------------------------------

def plot_song_predicted_emotions():
    path = DATA_CLEAN / "songs_with_predicted_emotions.csv"
    df = pd.read_csv(path)

    if "pred_emotion" in df.columns:
        counts = df["pred_emotion"].value_counts()
        labels = counts.index.tolist()
    elif "pred_emotion_id" in df.columns:
        counts = df["pred_emotion_id"].value_counts().sort_index()
        labels = [EMOTION_ID_TO_NAME.get(i, str(i)) for i in counts.index]
    else:
        raise ValueError(
            "Expected 'pred_emotion' or 'pred_emotion_id' column "
            f"in {path}"
        )

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=counts.values)
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Number of songs")
    plt.title("Predicted Emotion Distribution - Songs")
    plt.xticks(rotation=30)
    save_fig("song_predicted_emotions.png")


# -----------------------------------------------------------------------------
# 3. TF-IDF vocabulary size
# -----------------------------------------------------------------------------

def plot_tfidf_vocab_size():
    tfidf_path = MODELS_DIR / "tfidf_emotion.joblib"
    tfidf = joblib.load(tfidf_path)

    vocab_size = len(tfidf.vocabulary_)
    plt.figure(figsize=(4, 4))
    plt.bar(["TF-IDF vocab size"], [vocab_size])
    plt.ylabel("Number of features")
    plt.title("Emotion TF-IDF Vocabulary Size")
    for i, v in enumerate([vocab_size]):
        plt.text(i, v, str(v), ha="center", va="bottom")
    save_fig("tfidf_vocab_size.png")


# -----------------------------------------------------------------------------
# 4. Confusion matrix + 5. Classification report (using trained model)
# -----------------------------------------------------------------------------

def plot_confusion_and_classification():
    # Load data & model
    emotion_path = DATA_CLEAN / "emotion_clean.csv"
    df = pd.read_csv(emotion_path)

    if "emotion_id" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Expected 'emotion_id' and 'text' columns in {emotion_path}"
        )

    y_true = df["emotion_id"].values
    tfidf = joblib.load(MODELS_DIR / "tfidf_emotion.joblib")
    logreg = joblib.load(MODELS_DIR / "logreg_emotion.joblib")

    X = tfidf.transform(df["text"].astype(str).tolist())
    y_pred = logreg.predict(X)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = [EMOTION_ID_TO_NAME[i] for i in sorted(EMOTION_ID_TO_NAME.keys())]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Emotion Classification Confusion Matrix")
    save_fig("confusion_matrix.png")

    # Classification report: render as text in a matplotlib figure
    report_str = classification_report(
        y_true, y_pred, target_names=labels, digits=3
    )

    plt.figure(figsize=(8, 6))
    plt.axis("off")
    plt.text(
        0.01,
        0.05,
        report_str,
        fontsize=9,
        family="monospace",
    )
    plt.title("Emotion Classification Report", fontsize=12, pad=10)
    save_fig("classification_report.png")


# -----------------------------------------------------------------------------
# 6. Lyrics length histogram (clean_lyrics length)
# -----------------------------------------------------------------------------

def plot_lyrics_length_hist():
    path = DATA_CLEAN / "songs_with_predicted_emotions.csv"
    df = pd.read_csv(path)

    if "clean_lyrics" not in df.columns:
        raise ValueError(f"'clean_lyrics' column not found in {path}")

    lengths = df["clean_lyrics"].fillna("").str.len()

    plt.figure(figsize=(6, 4))
    plt.hist(lengths, bins=50)
    plt.xlabel("Length of clean_lyrics (characters)")
    plt.ylabel("Number of songs")
    plt.title("Distribution of Song Lyrics Lengths")
    save_fig("lyrics_length_hist.png")


# -----------------------------------------------------------------------------
# 7. Wordclouds per emotion (using songs_with_predicted_emotions.csv)
# -----------------------------------------------------------------------------

def plot_wordclouds():
    path = DATA_CLEAN / "songs_with_predicted_emotions.csv"
    df = pd.read_csv(path)

    if "pred_emotion" not in df.columns or "clean_lyrics" not in df.columns:
        print(
            "Skipping wordclouds: 'pred_emotion' or 'clean_lyrics' missing."
        )
        return

    emotions_to_plot = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    for emo in emotions_to_plot:
        subset = df[df["pred_emotion"] == emo]
        if subset.empty:
            print(f"No songs with emotion {emo}, skipping wordcloud.")
            continue

        text = " ".join(subset["clean_lyrics"].fillna("").tolist())
        if not text.strip():
            print(f"No lyrics text for emotion {emo}, skipping wordcloud.")
            continue

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=200,
        ).generate(text)

        plt.figure(figsize=(8, 4))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Wordcloud – {emo}")
        save_fig(f"wordcloud_{emo}.png")


# -----------------------------------------------------------------------------
# 8. Similarity heatmap for random subset of songs
# -----------------------------------------------------------------------------

def plot_similarity_heatmap(num_songs: int = 20, random_seed: int = 42):
    # Load song vectors + metadata
    path = DATA_CLEAN / "songs_with_predicted_emotions.csv"
    df = pd.read_csv(path)

    vectors_path = MODELS_DIR / "song_tfidf_matrix.npz"
    song_vectors = sparse.load_npz(vectors_path)

    n_songs = df.shape[0]
    if n_songs == 0:
        print("No songs found, skipping similarity heatmap.")
        return

    k = min(num_songs, n_songs)
    random.seed(random_seed)
    indices = sorted(random.sample(range(n_songs), k))

    sub_df = df.iloc[indices].reset_index(drop=True)
    sub_vecs = song_vectors[indices]

    sim_matrix = cosine_similarity(sub_vecs)

    # Use short labels: e.g., "Title (Artist)" truncated
    labels = []
    for _, row in sub_df.iterrows():
        title = str(row.get("title", "Unknown"))
        artist = str(row.get("artist", ""))
        label = f"{title} ({artist})"
        if len(label) > 20:
            label = label[:17] + "..."
        labels.append(label)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        sim_matrix,
        cmap="magma",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.title("Cosine Similarity Between Sampled Songs")
    save_fig("similarity_heatmap.png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Generating plots into:", RESULTS_DIR)

    plot_emotion_training_distribution()
    plot_song_predicted_emotions()
    plot_tfidf_vocab_size()
    plot_confusion_and_classification()
    plot_lyrics_length_hist()
    plot_wordclouds()
    plot_similarity_heatmap()

    print("All plots generated.")


if __name__ == "__main__":
    main()
