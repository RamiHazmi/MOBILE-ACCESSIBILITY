"""
backend/alarm_endpoint.py
══════════════════════════
WebSocket endpoint for real-time sound alarm detection (YAMNet).

Protocol:
  Phone → server (text):   "START" | "STOP"
  Phone → server (binary): PCM16 mono 16 kHz audio chunks (≈1.5 s each)

  Server → phone (JSON):
    {"type":"ready",     "status":"ok"|"no_model"}
    {"type":"status",    "message": str}
    {"type":"detection", "label":str, "emoji":str, "category":str,
                         "description":str, "confidence":float, "ts":str}
    {"type":"error",     "message": str}
"""

import json
import time
import asyncio
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool

import sound_alarm_service as sas

router = APIRouter()

# Preload YAMNet in a background thread so the first connection is fast
import threading
_loader = threading.Thread(target=sas.load_yamnet, daemon=True, name="yamnet-loader")
_loader.start()


@router.websocket("/ws/alarm")
async def alarm_websocket(ws: WebSocket):
    await ws.accept()
    print("[alarm_ws] ── client connected ──")

    # Respond immediately with current model state.
    # If not ready yet, the Flutter client will reconnect in ~8 s (retry loop).
    model_ok = sas.is_ready()
    print(f"[alarm_ws] model_ok={model_ok}")

    await ws.send_text(json.dumps({
        "type":   "ready",
        "status": "ok" if model_ok else "no_model",
    }))

    if not model_ok:
        await ws.send_text(json.dumps({
            "type":    "status",
            "message": "YAMNet still loading — reconnect in a few seconds",
        }))

    # ── Per-connection state ──────────────────────────────────────
    SAMPLE_RATE    = 16000
    BYTES_PER_SAMP = 2

    # Analyse every full chunk received (not a rolling window).
    # The Flutter client sends 1.5 s chunks, so each chunk IS the analysis window.
    audio_buf  = bytearray()
    monitoring = False
    chunk_idx  = 0

    # Cooldown: suppress same label within 3 seconds
    last_label:    str   = ""
    last_label_ts: float = 0.0
    COOLDOWN_SEC         = 3.0

    try:
        while True:
            msg = await ws.receive()

            # ── Text command ──────────────────────────────────────
            if "text" in msg and msg["text"]:
                cmd = msg["text"].strip().upper()
                print(f"[alarm_ws] cmd: {cmd}")

                if cmd == "START":
                    monitoring = True
                    audio_buf  = bytearray()
                    chunk_idx  = 0
                    await ws.send_text(json.dumps(
                        {"type": "status", "message": "Monitoring started"}))

                elif cmd == "STOP":
                    monitoring = False
                    await ws.send_text(json.dumps(
                        {"type": "status", "message": "Monitoring stopped"}))

            # ── Binary audio chunk ────────────────────────────────
            elif "bytes" in msg and msg["bytes"]:
                raw_bytes = msg["bytes"]
                print(f"[alarm_ws] received binary: {len(raw_bytes)} bytes  monitoring={monitoring}  model={model_ok}")

                if not monitoring or not model_ok:
                    continue

                chunk_idx += 1
                # Analyse each chunk directly — Flutter sends 1.5 s chunks
                detections = await run_in_threadpool(
                    sas.analyze, bytes(raw_bytes), SAMPLE_RATE)

                now = time.time()
                for det in detections:
                    label = det["label"]
                    if label == last_label and now - last_label_ts < COOLDOWN_SEC:
                        print(f"[alarm_ws] cooldown skip: {label}")
                        continue
                    last_label    = label
                    last_label_ts = now

                    payload = json.dumps({
                        "type":        "detection",
                        "label":       det["label"],
                        "emoji":       det["emoji"],
                        "category":    det["category"],
                        "description": det["description"],
                        "confidence":  det["confidence"],
                        "ts":          datetime.now().strftime("%H:%M"),
                    })
                    print(f"[alarm_ws] 🔔 sending detection: {payload}")
                    await ws.send_text(payload)

    except WebSocketDisconnect:
        print("[alarm_ws] client disconnected")
    except RuntimeError as exc:
        if "disconnect" in str(exc).lower():
            print("[alarm_ws] client disconnected (RuntimeError)")
        else:
            import traceback; traceback.print_exc()
    except Exception as exc:
        import traceback; traceback.print_exc()
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        print("[alarm_ws] connection closed")
