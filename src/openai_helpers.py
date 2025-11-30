# src/openai_helpers.py
from __future__ import annotations

"""
openai_helpers.py

Optional helper functions for enriching the user's mood text using OpenAI.

These helpers are *not required* for the core LyriSense pipeline:
    - Emotion classification uses a local TF-IDF + Logistic Regression model.
    - Recommendations use TF-IDF + cosine similarity.

If OPENAI_API_KEY is not set or any API error occurs, the functions fall back
to returning the original user text so the app continues to work normally.
"""

import os
from typing import Optional

from openai import OpenAI

# Cached OpenAI client (or None if no API key)
_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    """
    Return an OpenAI client if the API key exists, otherwise None.

    This avoids crashing your app if:
        - OPENAI_API_KEY is not set, or
        - You are running on a machine without network access.

    In those cases, any function that depends on OpenAI simply returns the
    original text untouched.
    """
    global _client
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    if _client is None:
        _client = OpenAI(api_key=api_key)

    return _client


def rewrite_feeling(text: str) -> str:
    """
    Use OpenAI to rewrite the user's feeling as a short, clear emotional
    description that your ML model can understand better.

    Example:
        Input:  "Everything feels heavy, but I’m trying to stay hopeful."
        Output: "I feel sad and overwhelmed, but still a bit hopeful."

    If anything fails (no key, API error, etc.), this function simply returns
    the original text so LyriSense still works.

    Args:
        text: Raw user input describing their mood.

    Returns:
        A rewritten emotional description, or the original text on failure.
    """
    client = _get_client()
    if client is None:
        # No API key set → just return original text, app still works
        return text

    prompt = (
        "The user is describing how they feel. "
        "Rewrite it in ONE OR TWO short sentences that clearly describe "
        "their emotional state in simple language (for example: "
        "'I feel sad and lonely but also a bit hopeful').\n\n"
        f"User: {text}\n\n"
        "Rewritten:"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # small, cheap, good enough for this
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You help clarify emotional descriptions. "
                        "Do NOT add explanations, just return the rewritten feeling."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        rewritten = resp.choices[0].message.content.strip()
        # Safety: fall back if something weird happens (empty string, etc.)
        return rewritten or text
    except Exception:
        # On any error, fail gracefully
        return text
