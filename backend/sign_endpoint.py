"""
backend/sign_endpoint.py
════════════════════════
WebSocket endpoint for real-time ASL sign language interpretation.

Uses the local CTR-GCN model (no Gemini/OpenRouter calls for recognition).
Groq is used only for BUILD_SENTENCE (optional, falls back gracefully).

Protocol:
  On connect, server sends {"type":"ready",...,"t_max":<int>} (model T_MAX from config).
  INTERPRET  {"cmd":"INTERPRET","mode":"word"|"letter","frames_b64":["…",...]}  (up to t_max JPEGs)
  BUILD_SENTENCE {"cmd":"BUILD_SENTENCE","signs":["HELLO","THANK YOU"]}
  RESET / PING

WebSocket URL:  ws://YOUR_PC_IP:8000/ws/sign
"""

import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sign_service import init_sign_service, get_service

router = APIRouter()

# Single worker keeps MediaPipe HolisticLandmarker on one OS thread forever.
_SIGN_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="asl_sign_infer",
)


@router.on_event("shutdown")
async def _shutdown_sign_executor():
    _SIGN_EXECUTOR.shutdown(wait=False)


@router.websocket("/ws/sign")
async def sign_websocket(ws: WebSocket):
    await ws.accept()
    print("[sign_ws] Client connected")

    svc = get_service()
    await ws.send_text(json.dumps({
        "type": "ready",
        "ok":   svc is not None,
        "model": "CTR-GCN (local)",
        "classes": len(svc._classes) if svc else 0,
        "t_max":  svc.t_max if svc is not None else 0,
    }))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await ws.send_text(json.dumps(
                    {"type": "error", "message": "Invalid JSON"}))
                continue

            cmd = msg.get("cmd", "").upper()

            # ── PING ──────────────────────────────────────────────
            if cmd == "PING":
                await ws.send_text(json.dumps(
                    {"type": "status", "message": "pong"}))

            # ── RESET ─────────────────────────────────────────────
            elif cmd == "RESET":
                await ws.send_text(json.dumps(
                    {"type": "status", "message": "Reset OK"}))

            # ── INTERPRET ─────────────────────────────────────────
            elif cmd == "INTERPRET":
                if svc is None:
                    await ws.send_text(json.dumps({
                        "type":    "error",
                        "message": "Model not loaded — check CTRGCN_MODEL_PATH",
                    }))
                    continue

                mode = msg.get("mode", "word").lower()

                # Accept multi-frame (new) or single-frame (legacy)
                frames_b64: list[str] = msg.get("frames_b64", [])
                if not frames_b64:
                    single = msg.get("frame_b64", "")
                    if single:
                        frames_b64 = [single]

                if not frames_b64:
                    await ws.send_text(json.dumps(
                        {"type": "error", "message": "No frames provided"}))
                    continue

                # Decode base64 → JPEG bytes (one full sequence, up to model T_MAX)
                cap = int(svc.t_max) if svc is not None else 0
                if cap < 1:
                    cap = 1
                jpeg_frames: list[bytes] = []
                for b64 in frames_b64[:cap]:
                    try:
                        jpeg_frames.append(base64.b64decode(b64))
                    except Exception:
                        pass

                if not jpeg_frames:
                    await ws.send_text(json.dumps(
                        {"type": "error", "message": "Bad base64 data"}))
                    continue

                n = len(jpeg_frames)
                await ws.send_text(json.dumps({
                    "type":    "status",
                    "message": f"Running CTR-GCN on {n} frame{'s' if n>1 else ''}…",
                }))

                loop = asyncio.get_running_loop()
                if mode == "letter":
                    result = await loop.run_in_executor(
                        _SIGN_EXECUTOR,
                        svc.interpret_letter,
                        jpeg_frames,
                    )
                    await ws.send_text(json.dumps({
                        "type":        "letter",
                        "letter":      result.get("letter", "?"),
                        "confidence":  result.get("confidence", 0.0),
                        "alternative": result.get("alternative"),
                        "note":        result.get("note"),
                    }))
                else:
                    result = await loop.run_in_executor(
                        _SIGN_EXECUTOR,
                        svc.interpret_word,
                        jpeg_frames,
                    )
                    await ws.send_text(json.dumps({
                        "type":         "word",
                        "sign":         result.get("sign", "UNCLEAR"),
                        "english":      result.get("english", ""),
                        "confidence":   result.get("confidence", 0.0),
                        "handshape":    result.get("handshape_observed", ""),
                        "alternatives": result.get("alternatives", []),
                        "note":         result.get("note"),
                    }))

            # ── BUILD_SENTENCE ────────────────────────────────────
            elif cmd == "BUILD_SENTENCE":
                if svc is None:
                    await ws.send_text(json.dumps(
                        {"type": "error", "message": "Model not loaded"}))
                    continue

                signs = msg.get("signs", [])
                if not signs:
                    await ws.send_text(json.dumps(
                        {"type": "error", "message": "No signs provided"}))
                    continue

                await ws.send_text(json.dumps({
                    "type":    "status",
                    "message": f"Building sentence from {len(signs)} signs…",
                }))

                result = await asyncio.get_running_loop().run_in_executor(
                    _SIGN_EXECUTOR,
                    svc.build_sentence,
                    signs,
                )
                await ws.send_text(json.dumps({
                    "type":       "sentence",
                    "sentence":   result.get("sentence", ""),
                    "confidence": result.get("confidence", 0.0),
                    "note":       result.get("note"),
                }))

            else:
                await ws.send_text(json.dumps(
                    {"type": "error", "message": f"Unknown command: {cmd}"}))

    except WebSocketDisconnect:
        print("[sign_ws] Client disconnected")
    except Exception as e:
        import traceback; traceback.print_exc()
        try:
            await ws.send_text(json.dumps(
                {"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        print("[sign_ws] Connection closed")