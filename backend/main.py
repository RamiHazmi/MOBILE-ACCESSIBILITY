"""
main.py — FastAPI backend (v4)
─────────────────────────────────
Endpoints
─────────
  GET  /                — health check
  POST /process         — voice → DialogueAgent → orchestrator
  POST /vision          — single camera frame → PerceptionAgent
  POST /navigate        — GPS update for an active trip
  POST /reset           — clear DialogueAgent's pending action
                          (used when user closes a screen)
  WS   /ws/vsr          — real-time lip reading (Chaplin VSR)

Security
────────
  • API key is read from OPENROUTER_API_KEY env var ONLY.
  • Hardcoded fallback key removed.
  • CORS open for development; tighten for production.

All upstream services used here are FREE:
  • OpenRouter :free models (no card required)
  • Whisper local STT (free, runs on CPU)
  • OpenStreetMap / Overpass / OSRM (free)
  • Groq (free tier, used by VSR corrector)
"""

import os
import sys

# Suppress verbose gRPC / MediaPipe / TensorFlow binary logging
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

import asyncio
import base64
import threading
import json
import re
import requests as _http

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PARENT_DIR)

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional

from agents.DialogueAgent      import DialogueAgent
from agents.orchestrator_graph import build_graph
from agents.StreetDangerAgent  import StreetDangerAgent
from agents.EmotionAgent       import EmotionAgent
from audio_utils import save_wav
from config      import (SAMPLE_RATE, OPENROUTER_API_KEY, GOOGLE_API_KEY,
                         LLM_MODEL_INTENT, LLM_MODEL_FALLBACKS,
                         YOLO_MODEL_PATH,
                         EMOTION_MODEL_PATH, EMOTION_CONF_THRESHOLD,
                         EMOTION_LABELS,
                         CTRGCN_MODEL_PATH, CTRGCN_CLASSES_PATH,
                         CTRGCN_NUM_FRAMES, GROQ_API_KEY)

# ── VSR (Chaplin lip-reading) + handwriting routers ───────────────
from vsr_endpoint import router as vsr_router
from handwriting_endpoint import router as handwriting_router


from sign_endpoint import router as sign_router
from gesture_endpoint import router as gesture_router
from alarm_endpoint import router as alarm_router          # WebSocket at /ws/alarm
from text_to_sign_endpoint import router as text_to_sign_router

app = FastAPI(title="Blind Glasses Backend v4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Static files (sign language MP4 videos) ───────────────────────
_static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(os.path.join(_static_dir, "signs", "letters"), exist_ok=True)
try:
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
    print("[Server] Static files mounted at /static")
except Exception as _e:
    print(f"[Server] Static mount skipped: {_e} (install aiofiles: pip install aiofiles)")

# ── Mount routers AFTER middleware ────────────────────────────────
app.include_router(handwriting_router)
app.include_router(vsr_router)         # WebSocket at /ws/vsr
app.include_router(sign_router)
app.include_router(gesture_router)     # WebSocket at /ws/gesture
app.include_router(alarm_router)       # WebSocket at /ws/alarm
app.include_router(text_to_sign_router)  # POST /sign/text_to_sign

# ── Preload Chaplin in background so first lip-reading is instant ─
from vsr_service import load_pipeline as _load_vsr
threading.Thread(target=_load_vsr, daemon=True).start()
print("[Server] VSR pipeline loading in background…")

# ── API key sanity check — warn loud and clear ──
if not OPENROUTER_API_KEY:
    print("[Server] WARNING: OPENROUTER_API_KEY env var not set.")
    print("[Server]   Voice intent (LLM fallback) will be disabled.")
    print("[Server]   Get a FREE key at https://openrouter.ai")
else:
    print(f"[Server] OpenRouter key loaded ({len(OPENROUTER_API_KEY)} chars)")

if not GOOGLE_API_KEY:
    print("[Server] WARNING: GOOGLE_API_KEY env var not set.")
    print("[Server]   Vision will fall back to OpenRouter (~150 req/day total).")
    print("[Server]   Get a FREE key at https://aistudio.google.com/apikey")
    print("[Server]   for 1000 vision requests/day on Gemini Flash-Lite.")
else:
    print(f"[Server] Google key loaded ({len(GOOGLE_API_KEY)} chars)")

dialogue = DialogueAgent(
    api_key=OPENROUTER_API_KEY,
    whisper_model_size="small",
    llm_model=LLM_MODEL_INTENT,
    llm_fallbacks=LLM_MODEL_FALLBACKS,
    google_api_key=GOOGLE_API_KEY,
)

# Local emotion classifier — silently disabled if model file is missing
emotion_agent = EmotionAgent(
    model_path=EMOTION_MODEL_PATH,
    labels=EMOTION_LABELS,
    conf_threshold=EMOTION_CONF_THRESHOLD,
)
print(f"[Server] EmotionAgent ready={emotion_agent.ready}")

graph, action, perception = build_graph(
    api_key=OPENROUTER_API_KEY,
    google_api_key=GOOGLE_API_KEY,
    emotion_agent=emotion_agent,
)
street_danger = StreetDangerAgent(model_path=YOLO_MODEL_PATH)
print(f"[Server] StreetDangerAgent ready={street_danger.ready}")

# ── Sign language services (CTR-GCN + MediaPipe Gesture) ─────────
from sign_service import init_sign_service
from gesture_service import init_gesture_service
from memory_service import (get_user_name, save_user_name,
                             save_personal_object as _mem_save_obj,
                             get_personal_object as _mem_get_obj,
                             get_all_personal_objects, delete_personal_object,
                             get_all_named_locations, delete_named_location,
                             get_all_sightings_for_user, delete_sighting,
                             save_personal_person as _mem_save_person,
                             get_personal_person as _mem_get_person,
                             get_all_personal_persons, delete_personal_person,
                             rename_personal_person, rename_personal_object,
                             rename_named_location,
                             relative_time)

try:
    init_sign_service(
        model_path   = CTRGCN_MODEL_PATH,
        classes_path = CTRGCN_CLASSES_PATH,
        num_frames   = CTRGCN_NUM_FRAMES,
        groq_api_key = GROQ_API_KEY,
    )
    print("[Server] SignService (CTR-GCN) ready")
except Exception as _e:
    print(f"[Server] SignService init failed: {_e}")

try:
    init_gesture_service(groq_api_key=GROQ_API_KEY)
    print("[Server] GestureService (ASL CNN) ready")
except Exception as _e:
    print(f"[Server] GestureService init failed: {_e}")

print("[Server] Ready — all models / APIs are FREE.")


# ─── Helpers ────────────────────────────────────────────────────

def _get_user_id(request: Request) -> str:
    """Read the device UUID sent by Flutter as X-User-ID header."""
    return (request.headers.get("X-User-ID") or "").strip()


def _err(detail: str) -> dict:
    return {
        "speak": "", "interrupt": False, "danger": "LOW", "mode": "IDLE",
        "found": None, "direction": None, "transcription": "",
        "intent": "unknown", "entities": [],
        "needs_clarification": False, "clarification_question": "",
        "awaiting_confirmation": False,
        "parse_error": True, "parse_error_detail": detail,
    }


def _mode_to_intent(mode: str) -> str:
    return {"FIND_OBJECT": "env_action",
            "DESCRIBE":    "env_action",
            "NAVIGATE":    "navigate"}.get(mode, "unknown")


def _build_state(
    intent, entities, language, transcription,
    frame_b64, mode, find_target,
    user_lat=None, user_lng=None,
    nav_destination=None, nav_mode=None,
    needs_clarification=False, clarification_question="",
    awaiting_confirmation=False,
    user_id="",
    force_replace=False,
) -> dict:
    return {
        "intent": intent, "entities": entities,
        "language": language, "transcription": transcription,
        "frame_b64": frame_b64,
        "scene": None, "danger_level": "LOW",
        "mode": mode, "find_target": find_target,
        "found": None, "direction": None,
        "user_lat": user_lat, "user_lng": user_lng,
        "nav_destination": nav_destination, "nav_mode": nav_mode,
        "nav_update": None,
        "speak": "", "interrupt": False,
        "needs_clarification":    needs_clarification,
        "clarification_question": clarification_question,
        "awaiting_confirmation":  awaiting_confirmation,
        "user_id": user_id,
        "_force_replace": force_replace,
    }


# ═══════════════════════════════════════════════════════════════
# /vision — used by camera_screen during a find/describe loop
# ═══════════════════════════════════════════════════════════════

@app.post("/vision")
async def vision(
    request:  Request,
    frame:    UploadFile = File(...),
    mode:     str        = Form("DESCRIBE"),
    target:   str        = Form(""),
    language: str        = Form("en"),
):
    try:
        frame_bytes = await frame.read()
        if len(frame_bytes) < 100:
            return {"speak": "", "interrupt": False, "danger": "LOW",
                    "found": False, "direction": None, "objects": []}

        frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
        state = _build_state(
            intent=_mode_to_intent(mode),
            entities=[target] if target else [],
            language=language,
            transcription=f"{mode.lower()} {target}".strip(),
            frame_b64=frame_b64, mode=mode,
            find_target=target or None,
            user_id=_get_user_id(request),
        )

        loop   = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, graph.invoke, state),
            timeout=20.0)

        scene   = result.get("scene") or {}
        objects = scene.get("objects", [])

        return {
            "speak":           result.get("speak", ""),
            "interrupt":       result.get("interrupt", False),
            "danger":          result.get("danger_level", "LOW"),
            "found":           result.get("found"),
            "direction":       result.get("direction"),
            "objects":         objects,
            "emotions":        result.get("emotions", []),
            "scene":           scene,
            "consensus":       scene.get("consensus_state", "SEARCHING"),
            "rate_limited":    result.get("rate_limited", False),
            "daily_exhausted": result.get("daily_exhausted", False),
        }
    except asyncio.TimeoutError:
        return {"speak": "", "interrupt": False, "danger": "LOW",
                "found": False, "direction": None, "objects": []}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"speak": "", "interrupt": False, "danger": "LOW",
                "found": False, "direction": None, "objects": []}


# ═══════════════════════════════════════════════════════════════
# /navigate — used by navigation_screen for GPS updates
# ═══════════════════════════════════════════════════════════════

@app.post("/navigate")
async def navigate_endpoint(
    request:     Request,
    lat:         float                = Form(...),
    lng:         float                = Form(...),
    destination: str                  = Form(""),
    nav_mode:    str                  = Form("walking"),
    language:    str                  = Form("en"),
    frame:       Optional[UploadFile] = File(None),
):
    try:
        frame_b64 = None
        if frame:
            frame_bytes = await frame.read()
            frame_b64   = base64.b64encode(frame_bytes).decode("utf-8")

        state = _build_state(
            intent="navigate",
            entities=[destination] if destination else [],
            language=language,
            transcription=(f"take me to {destination}" if destination else "navigate"),
            frame_b64=frame_b64, mode="NAVIGATE", find_target=None,
            user_lat=lat, user_lng=lng,
            nav_destination=destination or None, nav_mode=nav_mode,
            user_id=_get_user_id(request),
        )

        loop   = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, graph.invoke, state),
            timeout=25.0)

        nav_update = result.get("nav_update") or {}
        return {
            "speak":          result.get("speak", ""),
            "interrupt":      result.get("interrupt", False),
            "danger":         result.get("danger_level", "LOW"),
            "arrived":        nav_update.get("arrived", False),
            "off_route":      nav_update.get("off_route", False),
            "detour_warning": nav_update.get("detour_warning"),
            "current_step":   nav_update.get("current_step", 0),
            "dist_to_next_m": nav_update.get("dist_to_next_m"),
        }
    except asyncio.TimeoutError:
        return {"speak": "Navigation update timed out.", "interrupt": False,
                "danger": "LOW", "arrived": False, "off_route": False,
                "detour_warning": None}
    except Exception as e:
        import traceback; traceback.print_exc()
        return _err(str(e))


# ═══════════════════════════════════════════════════════════════
# /nav_danger — camera frames during navigation (YOLO + filter)
#               returns ONLY when there is something worth saying
# ═══════════════════════════════════════════════════════════════

@app.post("/nav_danger")
async def nav_danger(
    frame:    UploadFile = File(...),
    language: str        = Form("en"),
):
    try:
        if not street_danger.ready:
            return {
                "ready": False,
                "speak": "", "interrupt": False, "danger": "LOW",
                "objects": [], "dangers": [],
                "load_error": street_danger._load_error,
            }

        frame_bytes = await frame.read()
        if len(frame_bytes) < 100:
            return {"ready": True, "speak": "", "interrupt": False,
                    "danger": "LOW", "objects": [], "dangers": []}

        frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")

        loop   = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None,
                                 street_danger.detect_dangers,
                                 frame_b64,
                                 language),
            timeout=8.0)

        compact = []
        for o in result.get("objects", []):
            compact.append({
                "label":     o["label"],
                "conf":      round(o["conf"], 3),
                "bbox_xyxy": [round(v, 1) for v in o["bbox_xyxy"]],
                "img_w":     o["img_w"],
                "img_h":     o["img_h"],
                "proximity": o["proximity"],
                "direction": o["direction"],
                "is_danger": o["is_danger"],
            })

        return {
            "ready":     True,
            "speak":     result.get("speak", ""),
            "interrupt": result.get("interrupt", False),
            "danger":    result.get("danger", "LOW"),
            "objects":   compact,
            "dangers":   [c for c in compact if c["is_danger"]],
        }
    except asyncio.TimeoutError:
        return {"ready": True, "speak": "", "interrupt": False,
                "danger": "LOW", "objects": [], "dangers": []}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"ready": False, "speak": "", "interrupt": False,
                "danger": "LOW", "objects": [], "dangers": [],
                "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# /process — voice → DialogueAgent → orchestrator
# ═══════════════════════════════════════════════════════════════

@app.post("/process")
async def process(
    request:  Request,
    audio:    UploadFile           = File(...),
    frame:    Optional[UploadFile] = File(None),
    user_lat: Optional[float]      = Form(None),
    user_lng: Optional[float]      = Form(None),
):
    wav_path  = None
    frame_b64 = None
    try:
        audio_bytes = await audio.read()
        if len(audio_bytes) < 64:
            return _err("audio too small")

        try:
            wav_path = save_wav(audio_bytes, SAMPLE_RATE)
        except ValueError as e:
            return _err(str(e))

        loop = asyncio.get_event_loop()
        try:
            intent_dict = await asyncio.wait_for(
                loop.run_in_executor(None, dialogue.process_audio, wav_path),
                timeout=60.0)
        except asyncio.TimeoutError:
            return _err("Whisper timed out")

        print(f"[process] intent={intent_dict.get('intent')} "
              f"conf={intent_dict.get('confidence', 0):.2f} "
              f"src={intent_dict.get('source')} "
              f"awaiting={intent_dict.get('awaiting_confirmation')} "
              f"text='{intent_dict.get('transcription')}'")

        if frame:
            frame_bytes = await frame.read()
            frame_b64   = base64.b64encode(frame_bytes).decode("utf-8")

        needs_clar = intent_dict.get("needs_clarification", False)
        clar_q     = intent_dict.get("clarification_question", "")
        awaiting   = intent_dict.get("awaiting_confirmation", False)

        if (intent_dict.get("confirmed")
                and intent_dict.get("intent") == "env_action"
                and intent_dict.get("entities")):
            perception.reset_history(intent_dict["entities"][0])

        state = _build_state(
            intent=intent_dict.get("intent", "unknown"),
            entities=intent_dict.get("entities", []),
            language=intent_dict.get("language", "en"),
            transcription=intent_dict.get("transcription", ""),
            frame_b64=frame_b64, mode="IDLE", find_target=None,
            user_lat=user_lat, user_lng=user_lng,
            nav_destination=(
                (lambda e: e.get("value") or e.get("target") or str(e)
                 if isinstance(e, dict) else e)(
                    (intent_dict.get("entities") or [None])[0]
                )
                if intent_dict.get("intent") == "navigate"
                else None
            ),
            nav_mode="walking",
            needs_clarification=needs_clar,
            clarification_question=clar_q,
            awaiting_confirmation=awaiting,
            user_id=_get_user_id(request),
            force_replace=bool(intent_dict.get("_force_replace", False)),
        )

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, graph.invoke, state),
                timeout=60.0)
        except asyncio.TimeoutError:
            return _err("graph timed out")

        final_mode = result.get("mode", "IDLE")

        # Location already exists — set up voice YES/NO confirmation
        if final_mode == "CONFIRM_REPLACE_LOCATION":
            _label = ((result.get("entities") or intent_dict.get("entities") or [""])[0])
            _lang  = intent_dict.get("language", "en")
            dialogue.pending_action = {
                "intent":         "save_location",
                "entities":       [_label],
                "mode":           "SAVE_LOCATION",
                "language":       _lang,
                "_force_replace": True,
            }
            final_mode = "AWAIT_CONFIRM"

        if awaiting:
            final_mode = "AWAIT_CONFIRM"
        elif needs_clar:
            final_mode = "IDLE"
        # READ_HANDWRITING is passed through — Flutter opens HandwritingScreen
        # (mode stays "READ_HANDWRITING" — no override needed)

        return {
            "speak":                  result.get("speak", ""),
            "interrupt":              result.get("interrupt", False),
            "danger":                 result.get("danger_level", "LOW"),
            "mode":                   final_mode,
            "found":                  result.get("found"),
            "direction":              result.get("direction"),
            "scene":                  result.get("scene"),
            "transcription":          intent_dict.get("transcription", ""),
            "intent":                 intent_dict.get("intent", "unknown"),
            "entities":               intent_dict.get("entities", []),
            "language":               intent_dict.get("language", "en"),
            "needs_clarification":    needs_clar,
            "clarification_question": clar_q,
            "awaiting_confirmation":  awaiting,
            "confirmed":              intent_dict.get("confirmed", False),
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return _err(str(e))
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# /reset — clear pending action (called when user closes a screen)
# ═══════════════════════════════════════════════════════════════

@app.post("/reset")
async def reset():
    dialogue.reset_pending()
    perception.reset_history()
    street_danger.reset_cooldowns()
    action._last_text     = ""
    action._last_text_ts  = 0.0
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════
# /user/name — get / save the user's name for personalised greeting
# ═══════════════════════════════════════════════════════════════

@app.get("/user/name")
def get_name(request: Request):
    """Returns {"name": "<stored name or empty string>"}"""
    return {"name": get_user_name(_get_user_id(request))}


@app.post("/user/name")
async def set_name(http_req: Request, body: dict):
    """Body: {"name": "Ahmed"}  — saves the user's name."""
    name = (body.get("name") or "").strip()
    if not name:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": "name is required"})
    save_user_name(name, _get_user_id(http_req))
    return {"ok": True, "name": name}


# ═══════════════════════════════════════════════════════════════
# Personal object — save flow
# ═══════════════════════════════════════════════════════════════

@app.post("/describe_for_save")
async def describe_for_save_ep(
    request:  Request,
    frame:    UploadFile = File(...),
    name:     str        = Form("object"),
    language: str        = Form("en"),
):
    """
    VLM describes what's in the frame so the user can confirm before saving.
    Returns {description, speak} where speak includes the confirmation question.
    """
    uid         = _get_user_id(request)
    frame_bytes = await frame.read()
    if len(frame_bytes) < 100:
        return {"error": "no frame", "speak": ""}

    frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
    state = _build_state(
        intent="env_action", entities=[],
        language=language, transcription="describe",
        frame_b64=frame_b64, mode="DESCRIBE",
        find_target=None, user_id=uid,
    )

    try:
        loop   = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, graph.invoke, state),
            timeout=25.0)
        description = result.get("speak", "").strip()
    except Exception as e:
        print(f"[describe_for_save] error: {e}")
        description = ""

    if not description:
        _unclear = {
            "en": "I can't see clearly. Please tap to retake.",
            "fr": "Je ne vois pas clairement. Appuyez pour reprendre.",
            "ar": "لا أرى بوضوح. انقر لإعادة التقاط.",
            "tn": "ما نشوفش مليح. انقر باش تعيد.",
        }
        return {"description": "", "speak": _unclear.get(language, _unclear["en"])}

    # Check if a personal object with this name already exists for this user
    _already_warn = ""
    try:
        _existing_obj = _mem_get_obj(name.strip().lower(), uid)
        if _existing_obj:
            _already_warn = {
                "en": f"Note: You already have '{name}' saved. Tapping Save will replace it. ",
                "fr": f"Remarque : vous avez déjà '{name}' enregistré. Appuyer sur Sauvegarder le remplacera. ",
                "ar": f"ملاحظة: لديك '{name}' محفوظاً بالفعل. الحفظ سيستبدله. ",
                "tn": f"ملحوظة: عندك بالفعل '{name}' محفوظ. الحفظ باش يبدّله. ",
            }.get(language, f"Note: You already have '{name}' saved. Tapping Save will replace it. ")
    except Exception:
        pass

    _confirm = {
        "en": f"{_already_warn}{description} Is this your {name}? Tap once to save, double tap to retake.",
        "fr": f"{_already_warn}{description} C'est votre {name} ? Appuyez une fois pour sauvegarder, double appui pour reprendre.",
        "ar": f"{_already_warn}{description} هل هذا {name} الخاص بك؟ انقر مرة للحفظ، انقر مرتين لإعادة الالتقاط.",
        "tn": f"{_already_warn}{description} هذا {name} متاعك؟ انقر مرة باش تحفظ، انقر مرتين باش تعاود.",
    }
    return {
        "description":  description,
        "speak":        _confirm.get(language, _confirm["en"]),
        "name":         name,
        "already_exists": bool(_already_warn),
    }


@app.post("/describe_for_save_person")
async def describe_for_save_person_ep(
    request:      Request,
    frame:        UploadFile = File(...),
    person_name:  str        = Form("person"),
    relationship: str        = Form("person"),
    language:     str        = Form("en"),
):
    """VLM describes the face so the user can confirm before saving."""
    uid         = _get_user_id(request)
    frame_bytes = await frame.read()
    if len(frame_bytes) < 100:
        return {"error": "no frame", "speak": ""}

    frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")

    # Call VLM directly with a face-focused prompt (not the scene pipeline)
    try:
        loop = asyncio.get_event_loop()
        description = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                perception.describe_face,
                frame_b64, person_name, relationship,
            ),
            timeout=20.0,
        )
        description = (description or "").strip()
    except Exception as e:
        print(f"[describe_for_save_person] error: {e}")
        description = ""

    if not description:
        _unclear = {
            "en": "I can't see the face clearly. Please tap to retake.",
            "fr": "Je ne vois pas le visage clairement. Appuyez pour reprendre.",
            "ar": "لا أرى الوجه بوضوح. انقر لإعادة التقاط.",
            "tn": "ما نشوفش الوجه مليح. انقر باش تعيد.",
        }
        return {"description": "", "speak": _unclear.get(language, _unclear["en"])}

    _already_warn = ""
    try:
        _existing = _mem_get_person(person_name.strip().lower(), uid)
        if _existing:
            _already_warn = {
                "en": f"Note: You already have '{person_name}' saved. Saving will replace them. ",
                "fr": f"Remarque : vous avez déjà '{person_name}' enregistré. Sauvegarder le remplacera. ",
                "ar": f"ملاحظة: لديك '{person_name}' محفوظاً بالفعل. الحفظ سيستبدله. ",
                "tn": f"ملحوظة: عندك بالفعل '{person_name}' محفوظ. الحفظ باش يبدّله. ",
            }.get(language, f"Note: '{person_name}' already saved. Saving will replace. ")
    except Exception:
        pass

    _confirm = {
        "en": f"{_already_warn}{description} Is this {person_name}? Tap once to save, double tap to retake.",
        "fr": f"{_already_warn}{description} C'est {person_name} ? Appuyez une fois pour sauvegarder, double appui pour reprendre.",
        "ar": f"{_already_warn}{description} هل هذا {person_name}؟ انقر مرة للحفظ، انقر مرتين لإعادة الالتقاط.",
        "tn": f"{_already_warn}{description} هذا {person_name}؟ انقر مرة باش تحفظ، انقر مرتين باش تعاود.",
    }
    return {
        "description":    description,
        "speak":          _confirm.get(language, _confirm["en"]),
        "person_name":    person_name,
        "relationship":   relationship,
        "already_exists": bool(_already_warn),
    }


@app.post("/save_person")
async def save_person_ep(request: Request, body: dict):
    """Body: {name, relationship, description, image_b64?, language?}"""
    uid          = _get_user_id(request)
    name         = (body.get("name")         or "").strip()
    relationship = (body.get("relationship") or "person").strip()
    desc         = (body.get("description")  or "").strip()
    img          = (body.get("image_b64")    or "")
    lang         = (body.get("language")     or "en")

    if not name:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": "name required"})

    _mem_save_person(name, relationship, img, desc, uid)

    _saved = {
        "en": f"Saved! I'll recognize {name} as your {relationship} from now on.",
        "fr": f"Sauvegardé ! Je reconnais {name} comme votre {relationship} maintenant.",
        "ar": f"تم الحفظ! سأتعرف على {name} كـ{relationship} من الآن.",
        "tn": f"تحفظ! باش نعرف {name} كـ{relationship} من الآن.",
    }
    return {"speak": _saved.get(lang, _saved["en"]), "saved": True}


@app.post("/save_personal_object")
async def save_personal_object_ep(request: Request, body: dict):
    """Body: {name, description, image_b64?, language?} — saves to personal_objects DB."""
    uid  = _get_user_id(request)
    name = (body.get("name") or "").strip()
    desc = (body.get("description") or "").strip()
    img  = (body.get("image_b64") or "")
    lang = (body.get("language") or "en")

    if not name:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": "name required"})

    _mem_save_obj(name, desc, img, uid)

    _saved = {
        "en": f"Saved! I'll recognize {name} from now on. Say find my {name} anytime.",
        "fr": f"Sauvegardé ! Je reconnais {name} maintenant. Dites trouvez mon {name}.",
        "ar": f"تم الحفظ! سأتعرف على {name} الآن. قل ابحث عن {name} في أي وقت.",
        "tn": f"تحفظ! باش نعرف {name} من الآن. قول لقيلي {name} وقتاش ما تحب.",
    }
    return {"speak": _saved.get(lang, _saved["en"]), "saved": True}


# ═══════════════════════════════════════════════════════════════
# User profile — full read in one call
# ═══════════════════════════════════════════════════════════════

@app.get("/user/profile")
def get_user_profile(request: Request):
    """Returns all profile data for the device user in a single response."""
    uid       = _get_user_id(request)
    name      = get_user_name(uid)
    objects   = get_all_personal_objects(uid)
    persons   = get_all_personal_persons(uid)
    locations = get_all_named_locations(uid)
    sightings = get_all_sightings_for_user(uid, limit=40)
    for s in sightings:
        s["time_ago"] = relative_time(s.get("saved_at", ""), "en")
    return {
        "name":      name,
        "objects":   objects,
        "persons":   persons,
        "locations": locations,
        "history":   sightings,
    }


# ═══════════════════════════════════════════════════════════════
# User profile — delete operations
# ═══════════════════════════════════════════════════════════════

@app.delete("/user/persons/{name}")
def delete_user_person(name: str, request: Request):
    uid = _get_user_id(request)
    ok  = delete_personal_person(name, uid)
    return {"ok": ok}


@app.delete("/user/objects/{name}")
def delete_user_object(name: str, request: Request):
    uid = _get_user_id(request)
    ok  = delete_personal_object(name, uid)
    return {"ok": ok}


@app.delete("/user/locations/{label}")
def delete_user_location(label: str, request: Request):
    uid = _get_user_id(request)
    ok  = delete_named_location(label, uid)
    return {"ok": ok}


@app.delete("/user/history/{sighting_id}")
def delete_user_history(sighting_id: int, request: Request):
    uid = _get_user_id(request)
    ok  = delete_sighting(sighting_id, uid)
    return {"ok": ok}


# ═══════════════════════════════════════════════════════════════
# /profile/command — voice assistant for the profile page
# ═══════════════════════════════════════════════════════════════

def _fuzzy_best(query: str, candidates: list) -> str | None:
    """Return the closest candidate to query by edit distance, or None if too far."""
    if not candidates:
        return None
    q = query.lower().strip()
    for c in candidates:
        if c.lower() == q:
            return c
    for c in candidates:
        if q in c.lower() or c.lower() in q:
            return c

    def _lev(a: str, b: str) -> int:
        if len(a) > len(b):
            a, b = b, a
        cur = list(range(len(a) + 1))
        for j, cb in enumerate(b, 1):
            nxt = [j]
            for i, ca in enumerate(a, 1):
                nxt.append(min(cur[i] + 1, nxt[-1] + 1, cur[i - 1] + (0 if ca == cb else 1)))
            cur = nxt
        return cur[-1]

    best = min(candidates, key=lambda c: _lev(q, c.lower()))
    if _lev(q, best.lower()) <= max(2, len(q) // 3):
        return best
    return None


def _detect_lang_simple(text: str) -> str:
    ar_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
    if ar_chars > len(text) * 0.25:
        darija = ["نحب", "باش", "هك", "كيفاش", "شنو", "برشا", "ولا", "قرالي", "لقيلي"]
        return "tn" if any(m in text for m in darija) else "ar"
    tl = text.lower()
    fr = ["le ", "la ", "les ", "je ", "vous ", "mon ", "ma ", "mes ",
          "supprimer", "afficher", "lire", "montre"]
    if any(w in tl for w in fr):
        return "fr"
    return "en"


def _profile_build_speak(action, target_name, lang, objects, locations, sightings, persons=None):
    L = lang

    _persons = persons or []

    if action == "read_all":
        parts = []
        if _persons:
            pnames = ", ".join(f"{p['name']} ({p['relationship']})" for p in _persons)
            parts.append({
                "en": f"You have {len(_persons)} saved person{'s' if len(_persons)!=1 else ''}: {pnames}.",
                "fr": f"Vous avez {len(_persons)} personne(s) enregistrée(s) : {pnames}.",
                "ar": f"لديك {len(_persons)} شخص محفوظ: {pnames}.",
                "tn": f"عندك {len(_persons)} شخص محفوظ: {pnames}.",
            }.get(L, f"You have {len(_persons)} saved people: {pnames}."))
        if objects:
            names = ", ".join(o["name"] for o in objects)
            parts.append({
                "en": f"You have {len(objects)} saved object{'s' if len(objects)!=1 else ''}: {names}.",
                "fr": f"Vous avez {len(objects)} objet(s) enregistré(s) : {names}.",
                "ar": f"لديك {len(objects)} غرض محفوظ: {names}.",
                "tn": f"عندك {len(objects)} حاجة محفوظة: {names}.",
            }.get(L, f"You have {len(objects)} saved objects: {names}."))
        else:
            parts.append({"en": "No saved objects.", "fr": "Aucun objet.",
                          "ar": "لا توجد أغراض.", "tn": "ما عندكش حوايج."}.get(L, "No saved objects."))
        if locations:
            lnames = ", ".join(l["label"] for l in locations)
            parts.append({
                "en": f"{len(locations)} saved location{'s' if len(locations)!=1 else ''}: {lnames}.",
                "fr": f"{len(locations)} emplacement(s) : {lnames}.",
                "ar": f"{len(locations)} موقع محفوظ: {lnames}.",
                "tn": f"{len(locations)} موقع محفوظ: {lnames}.",
            }.get(L, f"{len(locations)} saved locations: {lnames}."))
        else:
            parts.append({"en": "No saved locations.", "fr": "Aucun emplacement.",
                          "ar": "لا توجد مواقع.", "tn": "ما عندكش مواقع."}.get(L, "No saved locations."))
        if sightings:
            recent = "; ".join(f"{s['object_name']} {s.get('time_ago','')}" for s in sightings[:5])
            parts.append({"en": f"Recent history: {recent}.", "fr": f"Historique récent : {recent}.",
                          "ar": f"السجل الأخير: {recent}.", "tn": f"السجل الأخير: {recent}."}.get(L, f"Recent: {recent}."))
        else:
            parts.append({"en": "No history yet.", "fr": "Aucun historique.",
                          "ar": "لا يوجد سجل.", "tn": "ما عندكش سجل."}.get(L, "No history."))
        return " ".join(parts)

    elif action == "read_objects":
        if not objects:
            return {"en": "You have no saved objects yet.",
                    "fr": "Aucun objet enregistré.",
                    "ar": "لا توجد أغراض محفوظة بعد.",
                    "tn": "ما عندكش حوايج محفوظة بعد."}.get(L, "No saved objects.")
        items = "; ".join(
            f"{o['name']}: {(o.get('description') or 'no description').strip()}" for o in objects)
        return {"en": f"Your objects: {items}.", "fr": f"Vos objets : {items}.",
                "ar": f"أغراضك: {items}.", "tn": f"حوايجك: {items}."}.get(L, f"Your objects: {items}.")

    elif action == "read_locations":
        if not locations:
            return {"en": "You have no saved locations yet.",
                    "fr": "Aucun emplacement enregistré.",
                    "ar": "لا توجد مواقع محفوظة بعد.",
                    "tn": "ما عندكش مواقع محفوظة بعد."}.get(L, "No saved locations.")
        items = ", ".join(l["label"] for l in locations)
        return {"en": f"Your saved locations: {items}.", "fr": f"Vos emplacements : {items}.",
                "ar": f"مواقعك: {items}.", "tn": f"مواقعك: {items}."}.get(L, f"Locations: {items}.")

    elif action == "read_history":
        if not sightings:
            return {"en": "No history yet. History builds as you find objects.",
                    "fr": "Aucun historique pour l'instant.",
                    "ar": "لا يوجد سجل بعد.",
                    "tn": "ما عندكش سجل بعد."}.get(L, "No history yet.")
        items = "; ".join(f"{s['object_name']} {s.get('time_ago','')}" for s in sightings[:7])
        return {"en": f"Your recent sightings: {items}.", "fr": f"Historique récent : {items}.",
                "ar": f"سجلاتك الأخيرة: {items}.", "tn": f"سجلاتك الأخيرة: {items}."}.get(L, f"Recent: {items}.")

    elif action == "read_people":
        if not _persons:
            return {"en": "You have no saved people yet. Say 'this is my friend [name]' to add someone.",
                    "fr": "Aucune personne enregistrée. Dites 'c'est mon ami [nom]' pour en ajouter.",
                    "ar": "لا يوجد أشخاص محفوظون. قل 'هذا صديقي [الاسم]' لإضافة شخص.",
                    "tn": "ما عندكش أشخاص محفوظين. قول 'هذا صديقي [الاسم]' باش تزيد شخص."}.get(L, "No saved people.")
        items = "; ".join(
            f"{p['name']} ({p['relationship']}): {(p.get('face_description') or 'no description').strip()[:60]}"
            for p in _persons)
        return {"en": f"Your saved people: {items}.", "fr": f"Vos personnes : {items}.",
                "ar": f"أشخاصك المحفوظون: {items}.", "tn": f"أشخاصك المحفوظين: {items}."}.get(L, f"Your people: {items}.")

    elif action in ("delete_object", "delete_location", "delete_history", "delete_person"):
        if not target_name:
            return {
                "en": "I couldn't understand which item to delete. Please say the name clearly.",
                "fr": "Je n'ai pas compris quel élément supprimer. Dites le nom clairement.",
                "ar": "لم أفهم أي عنصر تريد حذفه. قل الاسم بوضوح.",
                "tn": "ما فهمتش أشنو تحب تمسح. قول الاسم مليح.",
            }.get(L, "I couldn't understand which item to delete.")
        type_map = {
            "delete_object":   {"en": "object",      "fr": "objet",        "ar": "غرض",  "tn": "حاجة"},
            "delete_location": {"en": "location",    "fr": "emplacement",  "ar": "موقع", "tn": "موقع"},
            "delete_history":  {"en": "history for", "fr": "historique de","ar": "سجل",  "tn": "سجل"},
            "delete_person":   {"en": "person",      "fr": "personne",     "ar": "شخص",  "tn": "شخص"},
        }
        t = type_map.get(action, {}).get(L, "item")
        return {
            "en": f"Are you sure you want to delete the {t} '{target_name}'?",
            "fr": f"Voulez-vous vraiment supprimer le {t} '{target_name}' ?",
            "ar": f"هل أنت متأكد من حذف {t} '{target_name}'؟",
            "tn": f"واثق باش تمسح {t} '{target_name}'؟",
        }.get(L, f"Delete '{target_name}'?")

    else:
        return {
            "en": "Sorry, I can't do that here. You can list, delete, or rename your saved contacts, objects, and locations.",
            "fr": "Désolé, je ne peux pas faire ça ici. Vous pouvez lister, supprimer ou renommer vos contacts, objets et emplacements.",
            "ar": "آسف، لا أستطيع فعل ذلك هنا. يمكنك عرض أو حذف أو إعادة تسمية جهات الاتصال والأغراض والمواقع.",
            "tn": "آسف، ما نجمش نعمل هذا هنا. تنجم تقرا، تمسح، أو تبدّل اسم الأشخاص والحوايج والمواقع متاعك.",
        }.get(L, "Sorry, I can't do that here.")


_STOP_WORDS = {
    "the","my","a","an","delete","remove","erase","please","can","you","i",
    "want","to","it","this","that","some","just","also",
    "le","la","les","mon","ma","mes","supprimer","effacer","s'il","vous","plaît",
    "احذف","امسح","من","فضلك",
}


def _extract_noun(text: str, candidates: list) -> str | None:
    """
    Pull the most likely target noun from raw text.
    First tries fuzzy DB match; falls back to the last meaningful word.
    """
    words = [w.strip(".,!?\"'") for w in text.lower().split()
             if w.strip(".,!?\"'") not in _STOP_WORDS and len(w.strip(".,!?\"'")) > 1]
    if not words:
        return None
    for w in reversed(words):
        matched = _fuzzy_best(w, candidates)
        if matched:
            return matched
    # No DB match — return the most prominent word so the user sees what was heard
    return words[-1]


def _classify_profile_cmd(text, language, objects, locations, sightings,
                           pending_action: str = "", persons=None):
    """
    LLM + fallback classifier for profile page voice commands.

    pending_action: if the previous turn couldn't resolve a delete target, Flutter
    passes back the action string so we skip intent detection and focus entirely on
    identifying WHICH item the user wants to delete.
    """
    _persons   = persons or []
    obj_names  = [o["name"]   for o in objects]
    loc_labels = [l["label"]  for l in locations]
    sig_names  = list({s["object_name"] for s in sightings})
    per_names  = [p["name"]   for p in _persons]

    def _candidates_for(act):
        if act == "delete_object":   return obj_names
        if act == "delete_location": return loc_labels
        if act == "delete_history":  return sig_names
        if act == "delete_person":   return per_names
        return []

    # ── FOCUSED MODE: user is answering "which item?" ─────────────
    if pending_action in ("delete_object", "delete_location", "delete_history", "delete_person"):
        candidates = _candidates_for(pending_action)
        cand_str   = ", ".join(candidates) or "none"
        focused_prompt = (
            f'The user is completing a "{pending_action.replace("_"," ")}" command.\n'
            f'Available items: {cand_str}\n'
            f'User said: "{text}"\n\n'
            f'Identify which item they mean. '
            f'Understand by context, not just exact words — they may describe it, '
            f'say a nickname, or use a related word.\n'
            f'Respond ONLY with JSON: {{"target_name":"best match from the list"}}\n'
            f'- Fix Whisper transcription errors by matching to the closest item\n'
            f'- If nothing matches, return the key word(s) the user said\n'
            f'- NEVER return null'
        )
        target_name = None
        for _model in [LLM_MODEL_INTENT] + list(LLM_MODEL_FALLBACKS):
            try:
                resp      = _http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": _model,
                          "messages": [{"role": "user", "content": focused_prompt}],
                          "max_tokens": 30, "temperature": 0.05},
                    timeout=8,
                )
                rj = resp.json()
                if "choices" not in rj:
                    print(f"[profile_cmd] focused {_model}: {rj.get('error','no choices')}")
                    continue
                content = rj["choices"][0]["message"]["content"]
                m = re.search(r'\{.*?\}', content, re.DOTALL)
                if m:
                    target_name = json.loads(m.group()).get("target_name") or None
                break
            except Exception as _e:
                print(f"[profile_cmd] focused {_model} error: {_e}")

        # Fuzzy-match → DB name
        if target_name and candidates:
            target_name = _fuzzy_best(target_name, candidates) or target_name

        # Pure text extraction fallback
        if not target_name:
            target_name = _extract_noun(text, candidates)

        action   = pending_action
        in_db    = (target_name in candidates) if (target_name and candidates) else False
        return _build_delete_result(
            action, target_name, in_db, language,
            objects, locations, sightings, candidates)

    # ── NORMAL INTENT CLASSIFICATION ──────────────────────────────
    obj_ctx  = ", ".join(obj_names)  or "none"
    loc_ctx  = ", ".join(loc_labels) or "none"
    per_ctx  = ", ".join(f"{p['name']} ({p['relationship']})" for p in _persons) or "none"
    hist_ctx = ", ".join(
        f"{s['object_name']} ({s.get('time_ago','?')})" for s in sightings[:8]
    ) or "none"

    prompt = (
        f'You are an AI assistant for a blind user\'s profile page. '
        f'Understand the user\'s INTENT from context — do not pattern-match phrases.\n\n'
        f'Saved items:\n'
        f'PEOPLE: {per_ctx}\nOBJECTS: {obj_ctx}\nLOCATIONS: {loc_ctx}\nHISTORY: {hist_ctx}\n\n'
        f'User said: "{text}"\n\n'
        f'Respond ONLY with JSON (one object, no extra text):\n'
        f'{{"action":"read_all"|"read_people"|"read_objects"|"read_locations"|"read_history"'
        f'|"delete_person"|"delete_object"|"delete_location"|"delete_history"'
        f'|"rename_self"|"rename_person"|"rename_object"|"rename_location"|"unknown",'
        f'"target_name":"current name or null","new_name":"new desired name or null"}}\n\n'
        f'What each action means — judge by INTENT, not exact words:\n'
        f'- read_*: user wants to hear or know about saved items\n'
        f'- delete_*: user wants to remove an item permanently\n'
        f'- rename_self: user wants to update THEIR OWN name or identity — ANY phrasing:\n'
        f'  "my name is X", "I\'m X", "call me X", "change my name with X",\n'
        f'  "I want to be X", "I go by X", "actually I\'m X", "from now on X", etc.\n'
        f'  → new_name = X, target_name = null\n'
        f'- rename_person/object/location: user wants to rename a saved item\n'
        f'  → target_name = current saved name, new_name = desired new name\n'
        f'- unknown: request cannot be fulfilled here (calls, photos, navigation, weather…)\n\n'
        f'Rules:\n'
        f'- Extract new_name from the full sentence meaning, not just words after a connector\n'
        f'- Fix Whisper errors: match names to saved items by phonetic similarity\n'
        f'- new_name null only for non-rename actions\n'
        f'- Language: {language}'
    )

    action, target_name, new_name = "unknown", None, None

    # Try primary model then every fallback until one succeeds
    _models_to_try = [LLM_MODEL_INTENT] + list(LLM_MODEL_FALLBACKS)
    for _model in _models_to_try:
        try:
            resp      = _http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": _model,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 80, "temperature": 0.05},
                timeout=10,
            )
            resp_json = resp.json()
            if "choices" not in resp_json:
                _err_msg = resp_json.get("error", {})
                if isinstance(_err_msg, dict):
                    _err_msg = _err_msg.get("message", resp_json)
                print(f"[profile_cmd] {_model} → {_err_msg}")
                continue          # try next model
            content = resp_json["choices"][0]["message"]["content"]
            m = re.search(r'\{.*?\}', content, re.DOTALL)
            if m:
                parsed      = json.loads(m.group())
                action      = parsed.get("action", "unknown")
                target_name = parsed.get("target_name")
                new_name    = parsed.get("new_name")
            break                 # success — stop trying models
        except Exception as _e:
            print(f"[profile_cmd] {_model} exception: {_e}")

    # ── Emergency keyword fallback — only reached when ALL LLMs fail ─
    # This is intentionally minimal: detect broad category only.
    # Name extraction uses NO patterns — just strips filler words.
    if action == "unknown":
        tl  = text.lower()
        _del = any(w in tl for w in [
            "delete","remove","erase","supprimer","effacer","احذف","امسح","حذف"])
        _rd  = any(w in tl for w in [
            "list","show","read","tell","what","afficher","lire","اقرأ","اعرض","قرالي","وريني"])
        _pe  = any(w in tl for w in [
            "people","person","friend","family","personne","شخص","صديق","عائلة"])
        _lo  = any(w in tl for w in [
            "location","place","lieu","موقع","مكان","بلاصة"])
        _hi  = any(w in tl for w in [
            "history","histor","historique","سجل","تاريخ"])

        if _del:
            action = ("delete_person"   if _pe else
                      "delete_location" if _lo else
                      "delete_history"  if _hi else "delete_object")
        elif _rd:
            action = ("read_people"    if _pe else
                      "read_locations" if _lo else
                      "read_history"   if _hi else "read_all")
        # For rename: no keyword detection here — LLM must handle it.
        # If ALL models failed, we can't reliably extract names; tell the user to retry.

    # ── Fuzzy-match LLM target → real DB name ────────────────────
    if target_name:
        if   action == "delete_object":    target_name = _fuzzy_best(target_name, obj_names)  or target_name
        elif action == "delete_location":  target_name = _fuzzy_best(target_name, loc_labels) or target_name
        elif action == "delete_history":   target_name = _fuzzy_best(target_name, sig_names)  or target_name
        elif action == "delete_person":    target_name = _fuzzy_best(target_name, per_names)  or target_name
        elif action == "rename_person":    target_name = _fuzzy_best(target_name, per_names)  or target_name
        elif action == "rename_object":    target_name = _fuzzy_best(target_name, obj_names)  or target_name
        elif action == "rename_location":  target_name = _fuzzy_best(target_name, loc_labels) or target_name

    # ── Early return for rename (executed by profile_command) ─────
    if action.startswith("rename_"):
        return {"action": action, "target_name": target_name, "new_name": new_name,
                "target_ids": [], "speak": "", "confirm_required": False, "awaiting_target": False}

    # ── Last-resort noun extraction when LLM/fallback missed it ──
    if action.startswith("delete_") and not target_name:
        candidates = _candidates_for(action)
        target_name = _extract_noun(text, candidates)

    if not action.startswith("delete_"):
        # Read actions and unknown
        target_ids = []
        if action == "unknown":
            speak = _smart_fallback(text, language)
        else:
            speak = _profile_build_speak(action, None, language, objects, locations, sightings, _persons)
        return {"action": action, "target_name": None, "target_ids": target_ids,
                "speak": speak, "confirm_required": False, "awaiting_target": False}

    candidates = _candidates_for(action)
    in_db = (target_name in candidates) if (target_name and candidates) else False
    return _build_delete_result(
        action, target_name, in_db, language,
        objects, locations, sightings, candidates, persons=_persons)


def _build_delete_result(action, target_name, in_db, language,
                          objects, locations, sightings, candidates, persons=None):
    """Shared result builder for delete actions after target resolution."""
    L = language
    target_ids = []
    if action == "delete_history" and target_name and in_db:
        target_ids = [s["id"] for s in sightings if s["object_name"] == target_name]

    cat_labels = {
        "delete_object":   {"en":"object",   "fr":"objet",       "ar":"غرض",    "tn":"حاجة"},
        "delete_location": {"en":"location", "fr":"emplacement", "ar":"موقع",   "tn":"موقع"},
        "delete_history":  {"en":"history item","fr":"entrée",   "ar":"سجل",    "tn":"سجل"},
        "delete_person":   {"en":"person",   "fr":"personne",    "ar":"شخص",    "tn":"شخص"},
    }
    cat_labels_pl = {
        "delete_object":   {"en":"objects",   "fr":"objets",       "ar":"أغراض",  "tn":"حوايج"},
        "delete_location": {"en":"locations", "fr":"emplacements", "ar":"مواقع",  "tn":"مواقع"},
        "delete_history":  {"en":"history items","fr":"entrées",   "ar":"سجلات",  "tn":"سجلات"},
        "delete_person":   {"en":"people",    "fr":"personnes",    "ar":"أشخاص",  "tn":"أشخاص"},
    }
    sing = cat_labels.get(action, {}).get(L, "item")
    plur = cat_labels_pl.get(action, {}).get(L, "items")
    avail = ", ".join(candidates) if candidates else "none"

    if not target_name:
        # No target identified at all → ask which one
        speak = {
            "en": f"Which {sing} do you want to delete? You have: {avail}. Tap and say the name.",
            "fr": f"Quel {sing} voulez-vous supprimer ? Vous avez : {avail}. Appuyez et dites le nom.",
            "ar": f"أي {sing} تريد حذفه؟ لديك: {avail}. انقر وقل الاسم.",
            "tn": f"أشنو {sing} تحب تمسح؟ عندك: {avail}. دوس وقول الاسم.",
        }.get(L, f"Which {sing}? You have: {avail}.")
        return {"action": action, "target_name": None, "target_ids": [],
                "speak": speak, "confirm_required": False, "awaiting_target": True}

    if not in_db:
        # Target named but not in DB → list what they actually have
        speak = {
            "en": f"I don't have '{target_name}' in your saved {plur}. Your {plur}: {avail}. Tap and say the name.",
            "fr": f"Je n'ai pas '{target_name}' dans vos {plur}. Vos {plur} : {avail}. Appuyez et dites le nom.",
            "ar": f"ليس لديك '{target_name}' في {plur}. لديك: {avail}. انقر وقل الاسم.",
            "tn": f"ما عندكش '{target_name}' في {plur}. عندك: {avail}. دوس وقول الاسم.",
        }.get(L, f"I don't have '{target_name}'. Your {plur}: {avail}.")
        # Keep pending_action so user can immediately say the correct name
        return {"action": action, "target_name": None, "target_ids": [],
                "speak": speak, "confirm_required": False, "awaiting_target": True}

    # Happy path: target found in DB → ask for confirmation
    speak = _profile_build_speak(action, target_name, L, [], [], sightings, persons or [])
    return {"action": action, "target_name": target_name, "target_ids": target_ids,
            "speak": speak, "confirm_required": True, "awaiting_target": False}


def _smart_fallback(text: str, language: str) -> str:
    """
    When the user's intent cannot be mapped to a profile action,
    understand what they were trying to do and return a polite
    'Sorry, I can't [do that] here' message.
    """
    L = language
    default = {
        "en": ("Sorry, I can't do that on this page. "
               "Here you can list, delete, or rename your saved contacts, objects, and locations."),
        "fr": ("Désolé, je ne peux pas faire ça ici. "
               "Sur cette page vous pouvez : lister, supprimer ou renommer vos contacts, objets et emplacements."),
        "ar": ("آسف، لا أستطيع فعل ذلك هنا. "
               "في هذه الصفحة يمكنك عرض أو حذف أو إعادة تسمية جهات الاتصال والأغراض والمواقع."),
        "tn": ("آسف، ما نجمش نعمل هذا هنا. "
               "في هذه الصفحة تنجم تقرا، تمسح، أو تبدّل اسم الأشخاص والحوايج والمواقع متاعك."),
    }.get(L, "Sorry, I can't do that here.")

    if not OPENROUTER_API_KEY:
        return default

    _lang_names = {"en": "English", "fr": "French", "ar": "Arabic", "tn": "Tunisian Arabic"}
    p = (
        f'A user of a blind assistance app said: "{text}"\n'
        f'This is a profile management page. What the user CAN do here: '
        f'list saved contacts/objects/locations, delete any of them, '
        f'rename any of them, or change their own name.\n'
        f'Write 1-2 short, warm sentences:\n'
        f'1. Acknowledge exactly what they asked (e.g. "Sorry, I can\'t call your friend")\n'
        f'2. Briefly mention what they CAN do on this page\n'
        f'Language: {_lang_names.get(L, "English")}. Plain text only, no JSON.'
    )
    for _model in [LLM_MODEL_INTENT] + list(LLM_MODEL_FALLBACKS):
        try:
            resp      = _http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": _model,
                      "messages": [{"role": "user", "content": p}],
                      "max_tokens": 90, "temperature": 0.3},
                timeout=8,
            )
            resp_json = resp.json()
            if "choices" not in resp_json:
                print(f"[profile_cmd] smart_fallback {_model}: {resp_json.get('error','no choices')}")
                continue
            msg = resp_json["choices"][0]["message"]["content"].strip()
            return msg if msg else default
        except Exception as _e:
            print(f"[profile_cmd] smart_fallback {_model} error: {_e}")
    return default


@app.post("/profile/command")
async def profile_command(
    request:        Request,
    audio:          UploadFile = File(...),
    language:       str        = Form("en"),
    pending_action: str        = Form(""),
):
    """
    Voice command for the profile page.
    Whisper → LLM classification in context of user's actual DB data
    → fuzzy-match Whisper errors against real names → multilingual response.
    Returns {action, target_name, target_ids, speak, confirm_required, transcription, language}.
    """
    uid      = _get_user_id(request)
    wav_path = None
    _empty   = {"action": "unknown", "confirm_required": False,
                "target_name": None, "target_ids": [], "transcription": "", "language": language}
    try:
        audio_bytes = await audio.read()
        if len(audio_bytes) < 64:
            return {**_empty, "speak": {"en": "I didn't hear anything. Please try again.",
                                        "fr": "Je n'ai rien entendu. Réessayez.",
                                        "ar": "لم أسمع شيئاً. حاول مرة أخرى.",
                                        "tn": "ما سمعتش حاجة. عاود."}.get(language, "Please try again.")}
        try:
            wav_path = save_wav(audio_bytes, SAMPLE_RATE)
        except ValueError:
            return {**_empty, "speak": "Could not process audio."}

        loop = asyncio.get_event_loop()

        # 1. Transcribe with Whisper
        try:
            raw_text = await asyncio.wait_for(
                loop.run_in_executor(None, dialogue._transcribe, wav_path),
                timeout=30.0)
        except asyncio.TimeoutError:
            return {**_empty, "speak": "Speech recognition timed out."}

        if not raw_text or not raw_text.strip():
            return {**_empty, "speak": {"en": "I couldn't hear you clearly. Please try again.",
                                        "fr": "Je n'ai pas bien entendu. Réessayez.",
                                        "ar": "لم أسمعك بوضوح. حاول مرة أخرى.",
                                        "tn": "ما سمعتكش مليح. عاود."}.get(language, "Please try again.")}

        raw_text = raw_text.strip()
        # Refine language from the transcription text itself
        detected = _detect_lang_simple(raw_text)
        if detected != "en":
            language = detected

        # 2. Load user's profile for context
        objects   = get_all_personal_objects(uid)
        persons   = get_all_personal_persons(uid)
        locations = get_all_named_locations(uid)
        sightings = get_all_sightings_for_user(uid, limit=30)
        for s in sightings:
            s["time_ago"] = relative_time(s.get("saved_at", ""), language)

        # 3. Classify intent
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None, _classify_profile_cmd,
                    raw_text, language, objects, locations, sightings,
                    pending_action, persons),
                timeout=15.0)
        except asyncio.TimeoutError:
            result = {**_empty,
                      "speak": "Processing timed out. Please try again.",
                      "action": "unknown", "confirm_required": False,
                      "target_name": None, "target_ids": []}

        # 4. Execute rename immediately (no confirmation needed — it's reversible)
        _r_action = result.get("action", "")
        if _r_action.startswith("rename_"):
            _old = (result.get("target_name") or "").strip()
            _new = (result.get("new_name")    or "").strip()
            L    = language

            def _rn_speak(old, new):
                return {
                    "en": f"Done! {old.title()} is now called {new}.",
                    "fr": f"Fait ! {old.title()} s'appelle maintenant {new}.",
                    "ar": f"تم! {old} أصبح اسمه {new}.",
                    "tn": f"تم! {old} دروك اسمو {new}.",
                }.get(L, f"Done! Renamed to {new}.")

            def _rn_not_found(old, kind):
                return {
                    "en": f"I couldn't find '{old}' in your saved {kind}.",
                    "fr": f"Je n'ai pas trouvé '{old}' dans vos {kind}.",
                    "ar": f"ما لقيتش '{old}' في {kind} المحفوظ.",
                    "tn": f"ما لقيتش '{old}' في {kind} متاعك.",
                }.get(L, f"Couldn't find '{old}'.")

            if not _new:
                result["speak"] = {
                    "en": ("I heard you want to rename something. "
                           "Please say the full command, for example: 'rename Rami to Karim'."),
                    "fr": ("J'ai compris que vous voulez renommer quelque chose. "
                           "Dites la commande complète, par exemple : 'renommer Rami en Karim'."),
                    "ar": ("فهمت أنك تريد "
                           "إعادة التسمية. "
                           "قل الأمر كاملاً، "
                           "مثلاً: 'أعد تسمية "
                           "رامي إلى كريم'."),
                    "tn": ("فهمت باش تبدّل "
                           "اسم حاجة. قول الأمر "
                           "كامل، مثلاً: "
                           "'بدّل اسم رامي لـكريم'."),
                }.get(L, "Please say the full command, e.g. 'rename Rami to Karim'.")

            elif _r_action == "rename_self":
                save_user_name(_new, uid)
                result["speak"] = {
                    "en": f"Done! Your name is now {_new}.",
                    "fr": f"Fait ! Votre nom est maintenant {_new}.",
                    "ar": f"تم! اسمك الآن {_new}.",
                    "tn": f"تم! اسمك دروك {_new}.",
                }.get(L, f"Done! Your name is now {_new}.")

            elif _r_action == "rename_person":
                if not _old:
                    result["speak"] = {"en": "Which person do you want to rename?",
                                       "fr": "Quelle personne voulez-vous renommer ?",
                                       "ar": "أي شخص تريد إعادة تسميته؟",
                                       "tn": "أشنواه الشخص لي تحب تبدّل اسمو؟"
                                       }.get(L, "Which person?")
                else:
                    ok = rename_personal_person(_old, _new, uid)
                    result["speak"] = _rn_speak(_old, _new) if ok else _rn_not_found(_old, "people")

            elif _r_action == "rename_object":
                if not _old:
                    result["speak"] = {"en": "Which object do you want to rename?",
                                       "fr": "Quel objet voulez-vous renommer ?",
                                       "ar": "أي غرض تريد إعادة تسميته؟",
                                       "tn": "أشنواه الحاجة لي تحب تبدّل اسمها؟"
                                       }.get(L, "Which object?")
                else:
                    ok = rename_personal_object(_old, _new, uid)
                    result["speak"] = _rn_speak(_old, _new) if ok else _rn_not_found(_old, "objects")

            elif _r_action == "rename_location":
                if not _old:
                    result["speak"] = {"en": "Which location do you want to rename?",
                                       "fr": "Quel emplacement voulez-vous renommer ?",
                                       "ar": "أي موقع تريد إعادة تسميته؟",
                                       "tn": "أشنواه الموقع لي تحب تبدّل اسمو؟"
                                       }.get(L, "Which location?")
                else:
                    ok = rename_named_location(_old, _new, uid)
                    result["speak"] = _rn_speak(_old, _new) if ok else _rn_not_found(_old, "locations")

        return {**result, "transcription": raw_text, "language": language}

    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except Exception:
                pass


@app.delete("/user/history/object/{name}")
def delete_user_history_by_object(name: str, request: Request):
    """Delete ALL history entries for a given object name (scoped to user)."""
    uid      = _get_user_id(request)
    sightings = get_all_sightings_for_user(uid, limit=1000)
    deleted   = 0
    for s in sightings:
        if s.get("object_name", "").lower() == name.lower():
            if delete_sighting(s["id"], uid):
                deleted += 1
    return {"ok": deleted > 0, "deleted": deleted}


# ═══════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def health():
    return {
        "status":       "ok",
        "version":      "v4.1",
        "whisper":      dialogue.whisper is not None,
        "llm":          LLM_MODEL_INTENT,
        "vlm_primary":  perception.primary,
        "vlm_fallback": perception.fallback,
        "yolo_ready":   street_danger.ready,
        "yolo_path":    YOLO_MODEL_PATH,
        "maps":         "OpenStreetMap + Overpass + OSRM (FREE)",
        "api_key_set":  bool(OPENROUTER_API_KEY),
        "vsr_endpoint": "ws://<YOUR_IP>:8000/ws/vsr",
    }