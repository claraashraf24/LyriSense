# src/spotify_helpers.py
from __future__ import annotations

"""
spotify_helpers.py

Optional helper for enriching songs with Spotify metadata:
    - Spotify URL
    - 30-second preview URL
    - Cover image
    - Popularity score

This module is NOT required for the core LyriSense pipeline.
If SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are not set, the
functions will simply return empty metadata and the app will
continue to work without Spotify integration.
"""

import os
from typing import Optional, Dict

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# ---------------------------------------------------------------------------
# Global Spotipy client (or None if credentials are missing)
# ---------------------------------------------------------------------------

_client_id = os.getenv("SPOTIFY_CLIENT_ID")
_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

if _client_id and _client_secret:
    _spotify: Optional[spotipy.Spotify] = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=_client_id,
            client_secret=_client_secret,
        )
    )
else:
    _spotify = None  # Spotify integration disabled


def fetch_spotify_metadata(
    title: str,
    artist: str,
    market: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Search Spotify for the given title + artist and return basic metadata.

    If 'market' (e.g. 'US', 'CA', 'EG') is provided, Spotify will only return
    tracks available in that market. If the track is not available there,
    the result may be empty.

    If Spotify credentials are missing or any error occurs, the function
    returns a dict with None values and does NOT raise, so the rest of
    the app continues to work.

    Returns:
        {
            "spotify_url": str | None,
            "preview_url": str | None,
            "image_url": str | None,
            "spotify_popularity": int | None,
        }
    """
    # If Spotify isn’t configured, return empty fields gracefully
    if _spotify is None:
        return {
            "spotify_url": None,
            "preview_url": None,
            "image_url": None,
            "spotify_popularity": None,
        }

    query = f"track:{title} artist:{artist}"

    if market:
        market = market.upper()

    try:
        results = _spotify.search(q=query, type="track", limit=1, market=market)
    except Exception:
        # Any API error → fail gracefully
        return {
            "spotify_url": None,
            "preview_url": None,
            "image_url": None,
            "spotify_popularity": None,
        }

    items = results.get("tracks", {}).get("items", [])
    if not items:
        # Nothing found (or not available in that market)
        return {
            "spotify_url": None,
            "preview_url": None,
            "image_url": None,
            "spotify_popularity": None,
        }

    track = items[0]
    spotify_url = track.get("external_urls", {}).get("spotify")
    preview_url = track.get("preview_url")
    album = track.get("album", {}) or {}
    images = album.get("images", []) or []
    image_url = images[0]["url"] if images else None
    popularity = track.get("popularity")

    return {
        "spotify_url": spotify_url,
        "preview_url": preview_url,
        "image_url": image_url,
        "spotify_popularity": popularity,
    }
