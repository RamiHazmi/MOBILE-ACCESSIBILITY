"""
config.py — central configuration for the Blind Glasses backend.

ALL APIS / MODELS USED BELOW ARE 100% FREE:

  • Google Gemini API (free tier, no credit card)
      - gemini-2.5-flash-lite : 15 RPM, 1000 RPD ← BEST for the vision loop
      - gemini-2.5-flash      : 10 RPM,  250 RPD
      - gemini-2.5-pro        :  5 RPM,  100 RPD
      Resets at midnight Pacific Time.
      Get a free key: https://aistudio.google.com → "Get API key"

  • OpenRouter free-tier models (used as fallback after Gemini)
      ~50 req/day per :free model, resets at UTC midnight.

  • Whisper STT (faster-whisper) runs locally — free, offline.
  • Piper TTS (optional) runs locally — free, offline.
  • Nominatim + Overpass + OSRM (OpenStreetMap) — fully free, no API key.
"""

import os

# ─── Audio ──────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS    = 1

# ─── DialogueAgent / LLM (intent detection from voice) ──────────
# Keep OpenRouter for intent — it's only ~1 call per voice command,
# never burns the daily quota.  Gemini could work too but we don't
# want to compete with the vision loop for Gemini quota.
LLM_MODEL_INTENT = "meta-llama/llama-3.3-70b-instruct:free"
LLM_MODEL_FALLBACKS = [
    "openrouter/free",                          # auto-router
    "meta-llama/llama-3.1-8b-instruct:free",    # smaller, less throttled
    "google/gemma-3-12b-it:free",
    "nvidia/nemotron-nano-9b-v2:free",
]

# ─── PerceptionAgent / VLM (find object, describe scene) ────────
# Models prefixed with "gemini/" route to Google AI Studio.
# Models without that prefix route to OpenRouter.
#
# Strategy: Gemini Flash-Lite first because it has 1000 RPD (20× more
# than each OpenRouter free model), giving ~67 minutes of camera-on
# time per day from Google alone.  When Gemini is exhausted the chain
# falls back to OpenRouter for ~150 more requests.
#
# Total daily budget: ~1150 vision requests = ~75 min/day at 4-second
# scans, all free.
VLM_MODEL_VISION   = "gemini/gemini-2.5-flash-lite"   # PRIMARY (1000 RPD)

VLM_MODEL_FALLBACK = "gemini/gemini-2.5-flash"        # 2nd  (250 RPD)

VLM_MODEL_EXTRA_FALLBACKS = [
    "gemini/gemini-2.5-pro",                 # 3rd (100 RPD)
    "google/gemma-3-27b-it:free",            # 4th (OpenRouter, 50/day)
    "nvidia/nemotron-nano-12b-v2-vl:free",   # 5th (OpenRouter, 50/day)
    "openrouter/free",                       # 6th (OR auto-router)
]


# ─── API keys ───────────────────────────────────────────────────
GOOGLE_API_KEY     = os.environ.get("GOOGLE_API_KEY",     "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "").strip()

# ─── Vision tuning ──────────────────────────────────────────────
VISION_MIN_CONFIRMATIONS = 2     # frames in a row that must see target
VISION_MAX_HISTORY        = 3    # rolling window size
# Min seconds between successive VLM calls.  At 4s the daily Gemini
# Flash-Lite quota (1000 RPD) lasts ~67 min of continuous use.
# Don't push lower without realising you'll burn quota faster.
VISION_REQUEST_INTERVAL   = 4

# ─── Navigation tuning ──────────────────────────────────────────
NAV_PROXIMITY_RADIUS_KM   = 2.0
NAV_PROXIMITY_FALLBACK_KM = 8.0
NAV_OFF_ROUTE_THRESHOLD_M = 80
NAV_OFF_ROUTE_STRIKES     = 3

# ─── Confirmation flow ──────────────────────────────────────────
REQUIRE_CONFIRMATION = True

# ─── Street-danger detector (YOLO, free local) ──────────────────
# Used during navigation only — your fine-tuned best.pt detects
# street obstacles (cars, potholes, etc.).  Zero API calls.
# Paths resolved relative to this file so they work regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    os.path.join(_HERE, "models", "yolo", "best.pt"))
YOLO_CONF_THRESHOLD = float(os.environ.get(
    "YOLO_CONF_THRESHOLD", "0.45"))

EMOTION_MODEL_PATH = os.environ.get(
    "EMOTION_MODEL_PATH",
    os.path.join(_HERE, "models", "emotion", "best.h5"))
EMOTION_CONF_THRESHOLD = float(os.environ.get(
    "EMOTION_CONF_THRESHOLD", "0.45"))
# Order MUST match the order your model was trained with.  The
# default is the FER-2013 alphabetical convention.  If your model
# was trained with a different order, override here.
EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
]
# ─── CTR-GCN Sign Language Model ────────────────────────────────
CTRGCN_MODEL_PATH = os.environ.get(
    "CTRGCN_MODEL_PATH",
    os.path.join(_HERE, "models", "sign", "ctr-gcn"))   # ← DIRECTORY, not a .pt file
 
# CTRGCN_CLASSES_PATH is kept for backwards-compatibility but is now ignored;
# class names are read from class_names.json inside CTRGCN_MODEL_PATH.
CTRGCN_CLASSES_PATH = os.environ.get(
    "CTRGCN_CLASSES_PATH",
    os.path.join(_HERE, "models", "sign", "ctr-gcn", "class_names.json"))
 
CTRGCN_NUM_FRAMES = int(os.environ.get("CTRGCN_NUM_FRAMES", "60"))