# src/openai_helpers.py
from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

# Client will read OPENAI_API_KEY from env
_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    """
    Return an OpenAI client if the API key exists, otherwise None.
    This avoids crashing your app if the key is missing.
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
    Use OpenAI to rewrite the user's feeling as a short,
    clear emotional description that your ML model can understand better.

    If anything fails (no key, API error, etc.), it just returns the original text.
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
        # safety: fall back if something weird happens
        return rewritten or text
    except Exception:
        # On any error, fail gracefully
        return text
