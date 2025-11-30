from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model_loader import (
    predict_emotion,
    recommend_songs_from_text,
)

app = FastAPI(title="LyriSense API", version="0.1.0")

# CORS – allow frontend from anywhere (file://, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # no cookies, so keep this False with "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic models ----------

class EmotionRequest(BaseModel):
    text: str


class EmotionResponse(BaseModel):
    emotion_id: int
    emotion: str
    confidence: float


class RecommendRequest(BaseModel):
    text: str
    top_k: int = 5
    same_emotion_only: bool = True
    artist: Optional[str] = None       # optional artist filter
    sort_by: str = "similarity"        # "similarity" or "popularity"




class SongOut(BaseModel):
    title: str
    artist: str
    album: Optional[str] = None
    song_emotion: Optional[str] = None
    song_emotion_conf: float
    similarity: float
    lyric_snippet: Optional[str] = None
    popularity: Optional[float] = None






class TopEmotion(BaseModel):
    id: int
    name: str
    confidence: float


class RecommendResponse(BaseModel):
    user_emotion_id: int
    user_emotion: str
    user_confidence: float
    top2_emotions: List[TopEmotion]
    recommendations: List[SongOut]


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict", response_model=EmotionResponse)
def api_predict(req: EmotionRequest):
    try:
        pred = predict_emotion(req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return EmotionResponse(
        emotion_id=pred["emotion_id"],
        emotion=pred["emotion"],
        confidence=pred["confidence"],
    )


@app.post("/api/recommend", response_model=RecommendResponse)
def api_recommend(req: RecommendRequest):
    """
    Body must look like:
    {
      "text": "...",
      "top_k": 5,
      "same_emotion_only": true
    }
    """
    try:
        result = recommend_songs_from_text(
            user_text=req.text,
            top_k=req.top_k,
            same_emotion_only=req.same_emotion_only,
            artist_filter=req.artist,
            sort_by=req.sort_by,
        )


    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # result from similarity.recommend_songs_from_text:
    # {
    #   "user_emotion_id": ...,
    #   "user_emotion": ...,
    #   "user_confidence": ...,
    #   "top2_emotions": [(id, name, conf), ...],
    #   "recommendations": [ {title, artist, ...}, ... ],
    # }

    top2_list = [
        {"id": eid, "name": name, "confidence": conf}
        for (eid, name, conf) in result.get("top2_emotions", [])
    ]

    return RecommendResponse(
        user_emotion_id=result["user_emotion_id"],
        user_emotion=result["user_emotion"],
        user_confidence=result["user_confidence"],
        top2_emotions=top2_list,
        recommendations=[SongOut(**rec) for rec in result["recommendations"]],
    )
