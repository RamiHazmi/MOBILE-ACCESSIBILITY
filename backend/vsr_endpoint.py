"""
backend/vsr_endpoint.py
═══════════════════════
WebSocket endpoint for real-time lip reading.

Plug into main.py with TWO LINES:
    from vsr_endpoint import router as vsr_router
    app.include_router(vsr_router)

Protocol  (phone ↔ server)
─────────────────────────────
  Phone → server (binary):  JPEG frames continuously
  Phone → server (text):    "START_RECORDING" | "STOP_RECORDING" |
                             "PREDICT" | "RESET"

  Server → phone (text JSON):
    {"type":"ready",   "model": bool,  "error": str|null}
    {"type":"face",    "detected": bool, "lip_open": bool,
                       "buffered": int,  "lip_roi_b64": str|null}
    {"type":"result",  "raw": str, "final": str, "corrected": str,
                       "confidence": float, "lang": str,
                       "note": str|null, "frame_count": int,
                       "error": str|null}
    {"type":"status",  "message": str}
    {"type":"error",   "message": str}

WebSocket URL:  ws://YOUR_PC_IP:8000/ws/vsr
"""

import asyncio
import base64
import json
import time
from typing import List

import cv2
import numpy as np
import mediapipe as mp

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from vsr_service import (
    load_pipeline, is_ready, init_groq,
    run_vsr_on_frames, check_models,
    LIP_LANDMARKS, _LIP_TOP, _LIP_BOT, _LIP_L, _LIP_R,
)
import vsr_context

router = APIRouter()

# Init Groq corrector + context module once (key from config.py)
try:
    from config import GROQ_API_KEY
    init_groq(GROQ_API_KEY)
    vsr_context.init(GROQ_API_KEY)
except Exception as e:
    print(f"[vsr_endpoint] Groq init skipped: {e}")


# ─── MediaPipe wrapper (per-connection) ──────────────────────────
def _make_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode        = False,
        max_num_faces            = 1,
        refine_landmarks         = True,
        min_detection_confidence = 0.5,
    )


def _detect_lip(frame_bgr: np.ndarray, face_mesh) -> dict:
    """Same landmark logic as vsr_app.py preview loop."""
    h, w = frame_bgr.shape[:2]
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res  = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return {"detected": False, "lip_open": False, "roi_jpeg": None}

    lm = res.multi_face_landmarks[0].landmark
    xs = [lm[i].x * w for i in LIP_LANDMARKS]
    ys = [lm[i].y * h for i in LIP_LANDMARKS]
    lip_w = max(xs) - min(xs)
    lip_h = max(ys) - min(ys)

    # Generous padding so the full mouth is always visible
    x1 = max(0, int(min(xs)) - int(lip_w * 0.40))
    x2 = min(w, int(max(xs)) + int(lip_w * 0.40))
    y1 = max(0, int(min(ys)) - int(lip_h * 0.75))
    y2 = min(h, int(max(ys)) + int(lip_h * 0.75))

    # Aperture (vsr_app.py MIN_APERTURE_RATIO = 0.06)
    top_y, bot_y = lm[_LIP_TOP].y * h, lm[_LIP_BOT].y * h
    left_x, right_x = lm[_LIP_L].x * w, lm[_LIP_R].x * w
    aperture = (bot_y - top_y) / max(right_x - left_x, 1)
    lip_open = aperture > 0.06

    # Send normalized bbox so Flutter draws it on the camera preview
    bbox_norm = {
        "x": x1 / w, "y": y1 / h,
        "w": (x2 - x1) / w, "h": (y2 - y1) / h,
    }

    return {"detected": True, "lip_open": lip_open,
            "roi_jpeg": None, "bbox_norm": bbox_norm}


# ─── WebSocket handler ───────────────────────────────────────────
@router.websocket("/ws/vsr")
async def vsr_websocket(ws: WebSocket):
    await ws.accept()
    print("[vsr_ws] Client connected")

    m_ok, m_msg = check_models()
    await ws.send_text(json.dumps({
        "type":  "ready",
        "model": is_ready() or m_ok,
        "error": None if m_ok else m_msg,
    }))

    frame_buffer: List[np.ndarray] = []
    recording      = False
    face_mesh      = _make_face_mesh()
    last_face_msg  = 0.0

    # Lazy-load Chaplin (first time → ~30s on CPU)
    if not is_ready():
        await ws.send_text(json.dumps({
            "type": "status",
            "message": "Loading Chaplin model (first time, ~30s on CPU)..."
        }))
        await run_in_threadpool(load_pipeline)
        await ws.send_text(json.dumps({
            "type": "status",
            "message": "Model ready" if is_ready() else "Model load failed"
        }))

    try:
        while True:
            msg = await ws.receive()

            # ── Binary (JPEG frame) ───────────────────────────────
            if "bytes" in msg and msg["bytes"]:
                arr   = np.frombuffer(msg["bytes"], dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                if recording:
                    frame_buffer.append(frame.copy())

                # Throttle face messages to 5/sec
                now = time.time()
                if now - last_face_msg >= 0.2:
                    last_face_msg = now
                    info = _detect_lip(frame, face_mesh)
                    msg_out = {
                        "type":      "face",
                        "detected":  info["detected"],
                        "lip_open":  info["lip_open"],
                        "recording": recording,
                        "buffered":  len(frame_buffer),
                        "bbox_norm": info.get("bbox_norm"),
                    }
                    await ws.send_text(json.dumps(msg_out))

            # ── Text (control command) ────────────────────────────
            elif "text" in msg and msg["text"]:
                cmd = msg["text"].strip().upper()

                if cmd == "START_RECORDING":
                    recording    = True
                    frame_buffer = []
                    await ws.send_text(json.dumps({
                        "type": "status",
                        "message": "Recording started"}))

                elif cmd == "STOP_RECORDING":
                    recording = False
                    await ws.send_text(json.dumps({
                        "type": "status",
                        "message": "Recording stopped"}))

                elif cmd == "PREDICT":
                    recording = False
                    n = len(frame_buffer)
                    if n < 8:
                        await ws.send_text(json.dumps({
                            "type": "error",
                            "message": f"Too few frames ({n}). Hold longer."}))
                        continue

                    await ws.send_text(json.dumps({
                        "type": "status",
                        "message": f"Analysing {n} frames on CPU..."}))

                    frames_copy  = frame_buffer.copy()
                    frame_buffer = []

                    result = await run_in_threadpool(
                        run_vsr_on_frames, frames_copy)

                    # Run context analysis on the corrected text
                    context: dict = {}
                    final_text = result.get("final", "")
                    if final_text and not final_text.startswith("["):
                        try:
                            context = await run_in_threadpool(
                                vsr_context.analyze, final_text)
                        except Exception as _ce:
                            print(f"[vsr_endpoint] context analysis failed: {_ce}")

                    await ws.send_text(json.dumps({
                        "type":        "result",
                        "raw":         result["raw"],
                        "final":       result["final"],
                        "corrected":   result.get("corrected", ""),
                        "confidence":  result["confidence"],
                        "lang":        result["lang"],
                        "note":        result["note"],
                        "frame_count": result["frame_count"],
                        "error":       result["error"],
                        "intent":      context.get("intent", ""),
                        "actions":     context.get("actions", []),
                    }))

                elif cmd == "RESET":
                    recording    = False
                    frame_buffer = []
                    await ws.send_text(json.dumps({
                        "type": "status",
                        "message": "Reset OK"}))

    except WebSocketDisconnect:
        print("[vsr_ws] Client disconnected")
    except RuntimeError as e:
        if "disconnect" in str(e).lower():
            print("[vsr_ws] Client disconnected (RuntimeError)")
        else:
            import traceback; traceback.print_exc()
    except Exception as e:
        import traceback; traceback.print_exc()
        try:
            await ws.send_text(json.dumps({
                "type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        face_mesh.close()
        print("[vsr_ws] Connection closed")


# ─── REST: translation ───────────────────────────────────────────
class _TranslateBody(BaseModel):
    text: str
    target_language: str  # french | arabic | darija


@router.post("/vsr/translate")
async def vsr_translate(body: _TranslateBody):
    try:
        translation = await run_in_threadpool(
            vsr_context.translate, body.text, body.target_language)
        return {"translation": translation, "language": body.target_language}
    except Exception as e:
        return {"translation": "", "error": str(e)}