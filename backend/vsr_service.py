"""
backend/vsr_service.py
═══════════════════════
Chaplin VSR wrapped as a singleton service for FastAPI.

FIX: write grayscale MP4 (isColor=False) — identical to what the working
     Streamlit app does.  The previous XVID / "3-channel gray" trick caused
     Chaplin's internal MediaPipe detector to fail silently → empty result.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import threading
import unittest.mock
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# ─── Chaplin path setup ───────────────────────────────────────────
CHAPLIN_DIR = Path(__file__).parent / "chaplin"
CONFIG_FILE = str(CHAPLIN_DIR / "configs" / "LRS3_V_WER19.1.ini")

os.chdir(str(CHAPLIN_DIR))

for p in [str(CHAPLIN_DIR), str(CHAPLIN_DIR / "espnet")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Mock chainer
for mod in ["chainer", "chainer.backends", "chainer.backends.cuda"]:
    if mod not in sys.modules:
        sys.modules[mod] = unittest.mock.MagicMock()

# ─── Pipeline constants ───────────────────────────────────────────
# FIX: use 16 fps — same as the working Streamlit app.
# Chaplin resamples internally; the key thing is codec compatibility.
TARGET_FPS        = 16
FRAME_COMPRESSION = 25
_OUT_W, _OUT_H    = 640 // 3, 480 // 3   # 213 × 160

LIP_LANDMARKS = [
    61, 146,  91, 181,  84,  17, 314, 405, 321, 375, 291,
    78,  95,  88, 178,  87,  14, 317, 402, 318, 324, 308,
    13, 312, 311, 310, 415, 191,  80,  81,  82,
]
_LIP_TOP, _LIP_BOT = 13, 17
_LIP_L,   _LIP_R   = 61, 291

# Debug: save last segment here so you can inspect it manually
DEBUG_SAVE_DIR = Path(__file__).parent / "vsr_debug"


# ─── Model file check ────────────────────────────────────────────
def check_models() -> tuple[bool, str]:
    files = [
        CHAPLIN_DIR / "benchmarks" / "LRS3" / "models" /
            "LRS3_V_WER19.1" / "model.pth",
        CHAPLIN_DIR / "benchmarks" / "LRS3" / "models" /
            "LRS3_V_WER19.1" / "model.json",
    ]
    for f in files:
        if not f.exists():
            return False, f"Missing: {f}"
    return True, "OK"


# ─── Pipeline singleton ──────────────────────────────────────────
_pipeline      = None
_pipeline_lock = threading.Lock()
_pipeline_ok   = False
_pipeline_err  = ""


def load_pipeline():
    global _pipeline, _pipeline_ok, _pipeline_err
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        try:
            from pipelines.pipeline import InferencePipeline
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"[vsr_service] Loading Chaplin on {device}...")
            _pipeline = InferencePipeline(
                config_filename = CONFIG_FILE,
                detector        = "mediapipe",
                face_track      = True,
                device          = device,
            )
            _pipeline_ok = True
            print("[vsr_service] Chaplin loaded OK")
        except Exception as e:
            _pipeline_err = str(e)
            print(f"[vsr_service] Chaplin load FAILED: {e}")
        return _pipeline


def is_ready() -> bool:
    return _pipeline is not None and _pipeline_ok


# ─── Groq corrector ───────────────────────────────────────────────
class GroqCorrector:
    SYSTEM = """You are correcting lip-reading transcriptions from Auto-AVSR (LRS3 model).

The model:
- Was trained ONLY on English (LRS3 / BBC interviews)
- Outputs ALL-CAPS English with phonetic errors
- May hallucinate when conditions are not ideal

Your job:
1. Convert ALL-CAPS to natural capitalization
2. Fix obvious phonetic errors with common English words
3. Add proper punctuation
4. If the output looks like nonsense or hallucination, return "[unclear]"
5. Stay close to the input. Do NOT invent content.

Common English words to bias toward: hello, hi, thank you, yes, no, please,
sorry, my name is, how are you, what, where, when, today, tomorrow, friend.

Return JSON only:
{
  "detected_lang": "en",
  "corrected": "corrected text or [unclear]",
  "final_output": "final text or [unclear]",
  "confidence": 0.0-1.0,
  "note": null or short reason
}"""

    def __init__(self, api_key: str):
        from groq import Groq
        self.client = Groq(api_key=api_key)

    def correct(self, raw: str) -> dict:
        if not raw.strip() or raw.startswith("[ERR"):
            return {"detected_lang": "?", "corrected": raw,
                    "final_output": raw, "confidence": 0.0,
                    "note": "Invalid input"}
        try:
            resp = self.client.chat.completions.create(
                model    = "llama-3.3-70b-versatile",
                messages = [
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user",
                     "content": f'Raw VSR transcription: "{raw}"'},
                ],
                temperature = 0.1,
                max_tokens  = 512,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            return {"detected_lang": "?", "corrected": raw,
                    "final_output": raw, "confidence": 0.0,
                    "note": str(e)}


_groq: Optional[GroqCorrector] = None

def init_groq(api_key: str):
    global _groq
    if api_key and not _groq:
        try:
            _groq = GroqCorrector(api_key)
            print("[vsr_service] Groq corrector ready")
        except Exception as e:
            print(f"[vsr_service] Groq init failed: {e}")


# ─── Write frames to grayscale MP4 ───────────────────────────────
def _write_video_mp4(path: str, frames: list) -> bool:
    """
    FIX: Write BGR frames as a TRUE grayscale MP4 (isColor=False).

    This matches exactly what the working Streamlit app does:
        vout = cv2.VideoWriter(vid_p, cv2.VideoWriter_fourcc(*"mp4v"),
                               TARGET_FPS, (ow, oh), False)   ← isColor=False!
        gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        vout.write(gray)

    The previous XVID / "3-channel gray" approach caused Chaplin's internal
    MediaPipe face-mesh detector to produce no landmarks, returning empty text.
    """
    out = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        TARGET_FPS,
        (_OUT_W, _OUT_H),
        isColor=False,      # ← KEY FIX: grayscale, matching Streamlit
    )

    if not out.isOpened():
        print("[vsr_service] VideoWriter (mp4v grayscale) failed to open")
        return False

    written = 0
    enc_params = [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_COMPRESSION]

    for f in frames:
        # Front camera frames are mirrored; flip so MediaPipe sees natural
        # left/right orientation (same as the laptop webcam which isn't mirrored)
        f = cv2.flip(f, 1)

        # Resize to Chaplin's expected input size
        small = cv2.resize(f, (_OUT_W, _OUT_H), interpolation=cv2.INTER_AREA)

        # FIX: encode as JPEG then decode as grayscale — identical to Streamlit
        _, buf = cv2.imencode('.jpg', small, enc_params)
        gray   = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

        out.write(gray)
        written += 1

    out.release()
    print(f"[vsr_service] wrote {written} grayscale frames → {path}")
    return written > 0


# ─── Core inference ───────────────────────────────────────────────
def run_vsr_on_frames(frames: list) -> dict:
    if not frames:
        return _vsr_error("No frames received")

    n = len(frames)
    print(f"[vsr_service] run_vsr_on_frames: {n} frames received")

    pipeline = load_pipeline()
    if pipeline is None:
        return _vsr_error(f"Pipeline not loaded: {_pipeline_err}")

    tmp_d = tempfile.mkdtemp(prefix="vsr_ws_")
    try:
        vid_path = os.path.join(tmp_d, "segment.mp4")   # MP4 — matches Streamlit

        ok = _write_video_mp4(vid_path, frames)
        if not ok:
            return _vsr_error("VideoWriter produced 0 frames")

        # Verify file size — if tiny, codec failed silently
        fsize = os.path.getsize(vid_path)
        print(f"[vsr_service] video file size: {fsize} bytes")
        if fsize < 2048:
            return _vsr_error(f"Video file too small ({fsize}B) — codec issue")

        # Save a debug copy so you can play it and check visually
        try:
            DEBUG_SAVE_DIR.mkdir(exist_ok=True)
            debug_path = str(DEBUG_SAVE_DIR / "last_segment.mp4")
            shutil.copy2(vid_path, debug_path)
            print(f"[vsr_service] debug copy → {debug_path}")
        except Exception:
            pass

        t0  = time.time()
        raw = pipeline(vid_path) or ""
        dt  = time.time() - t0
        print(f"[vsr_service] inference {dt:.1f}s  raw='{raw[:80]}'")

        if not raw.strip():
            return {"raw": "", "final": "", "corrected": "",
                    "confidence": 0.0, "lang": "?",
                    "note": "No speech detected",
                    "error": None, "frame_count": n}

        if _groq:
            res = _groq.correct(raw)
        else:
            res = {"final_output": raw, "corrected": raw,
                   "confidence": 0.0, "detected_lang": "en", "note": None}

        return {
            "raw":         raw,
            "final":       res.get("final_output", raw),
            "corrected":   res.get("corrected", raw),
            "confidence":  res.get("confidence", 0.0),
            "lang":        res.get("detected_lang", "en"),
            "note":        res.get("note"),
            "error":       None,
            "frame_count": n,
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return _vsr_error(str(e))
    finally:
        shutil.rmtree(tmp_d, ignore_errors=True)


def _vsr_error(msg: str) -> dict:
    return {"raw": "", "final": f"[Error] {msg}", "corrected": "",
            "confidence": 0.0, "lang": "?", "note": msg,
            "error": msg, "frame_count": 0}