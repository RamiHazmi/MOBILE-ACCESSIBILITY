"""
gesture_endpoint.py
═══════════════════
WebSocket endpoint for MediaPipe Gesture Recognizer.

Drop-in replacement for sign_endpoint.py — identical wire protocol so
sign_language_page.dart works with zero changes.

WebSocket URL:  ws://YOUR_PC_IP:8000/ws/gesture

Mount in main.py with TWO lines (do not touch anything else):
    from gesture_endpoint import router as gesture_router
    app.include_router(gesture_router)

The Flutter sign_language_page.dart only needs its WS URL changed from
    /ws/sign  →  /ws/gesture
which is done in gesture_sign_page.dart (new file, original untouched).
"""

import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from gesture_service import init_gesture_service, get_gesture_service

router = APIRouter()

_GESTURE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="gesture_infer",
)


@router.on_event("shutdown")
async def _shutdown():
    _GESTURE_EXECUTOR.shutdown(wait=False)


@router.websocket("/ws/gesture")
async def gesture_websocket(ws: WebSocket):
    await ws.accept()
    print("[gesture_ws] Client connected")

    svc = get_gesture_service()
    await ws.send_text(json.dumps({
        "type":    "ready",
        "ok":      svc is not None,
        "model":   "SigLIP ViT (prithivMLmods/Alphabet-Sign-Language-Detection)",
        "classes": 26,
        "t_max":   svc.t_max if svc is not None else 20,
    }))

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps(
                    {"type": "status", "message": "pong"}))
                continue
            except Exception:
                break

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
                        "message": "Gesture model not loaded",
                    }))
                    continue

                mode = msg.get("mode", "word").lower()

                frames_b64: list[str] = msg.get("frames_b64", [])
                if not frames_b64:
                    single = msg.get("frame_b64", "")
                    if single:
                        frames_b64 = [single]

                if not frames_b64:
                    await ws.send_text(json.dumps(
                        {"type": "error", "message": "No frames provided"}))
                    continue

                jpeg_frames: list[bytes] = []
                for b64 in frames_b64:
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
                    "message": f"Recognising gesture in {n} frame{'s' if n>1 else ''}…",
                }))

                loop = asyncio.get_running_loop()

                if mode == "letter":
                    result = await loop.run_in_executor(
                        _GESTURE_EXECUTOR,
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
                        _GESTURE_EXECUTOR,
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
                    "message": f"Building sentence from {len(signs)} gestures…",
                }))

                result = await asyncio.get_running_loop().run_in_executor(
                    _GESTURE_EXECUTOR,
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
        print("[gesture_ws] Client disconnected")
    except Exception as e:
        import traceback; traceback.print_exc()
        try:
            await ws.send_text(json.dumps(
                {"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        print("[gesture_ws] Connection closed")
