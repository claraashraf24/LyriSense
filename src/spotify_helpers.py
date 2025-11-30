from __future__ import annotations

import os
from typing import Optional, Dict

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Initialize Spotipy client once (global)
_client_id = os.getenv("SPOTIFY_CLIENT_ID")
_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

if not _client_id or not _client_secret:
    raise RuntimeError(
        "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set as environment variables."
    )

_spotify = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=_client_id,
        client_secret=_client_secret,
    )
)


def fetch_spotify_metadata(
    title: str,
    artist: str,
    market: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Search Spotify for the given title + artist.

    If 'market' (e.g. 'US', 'CA', 'EG') is provided, Spotify will only return
    tracks available in that market. If the track is *not* available there,
    the result will be empty.
    """
    query = f"track:{title} artist:{artist}"

    # 'market' can be None or a 2-letter code like 'US', 'EG', 'CA'
    if market:
        market = market.upper()

    results = _spotify.search(q=query, type="track", limit=1, market=market)

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
