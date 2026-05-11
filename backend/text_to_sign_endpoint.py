"""
text_to_sign_endpoint.py
────────────────────────
POST /sign/text_to_sign
  Body (JSON): { "text": "Hello, how are you?", "remove_stopwords": false }
  Returns:     { "tokens": [...], "segments": [...] }
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from text_to_sign_service import build_sign_sequence, text_to_tokens

router = APIRouter()


class _Body(BaseModel):
    text: str
    remove_stopwords: bool = False


@router.post("/sign/text_to_sign")
async def text_to_sign(req: Request, body: _Body):
    base_url = str(req.base_url).rstrip("/")
    segments = build_sign_sequence(
        body.text,
        base_url=base_url,
        remove_stopwords=body.remove_stopwords,
    )
    tokens = [s["token"] for s in segments]
    return JSONResponse({"tokens": tokens, "segments": segments})
