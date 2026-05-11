"""
gesture_service.py
══════════════════
ASL Alphabet — HuggingFace SigLIP Vision Transformer.

Model : prithivMLmods/Alphabet-Sign-Language-Detection
        SiglipForImageClassification, 224×224 RGB input.
        26 classes A–Z (includes J and Z).
        99.96% accuracy on sign language alphabet benchmark.

Downloaded automatically to HuggingFace cache on first run.

Hand ROI strategy:
  1. Try MediaPipe Hands to get a tight hand bounding box.
  2. Fall back to center-square crop of the frame.

t_max = 20  — Flutter captures 20 frames (~3 s); majority vote picks winner.
"""

from __future__ import annotations

import io
import json
import threading
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_MODEL_ID = "prithivMLmods/Alphabet-Sign-Language-Detection"

# ── MediaPipe Hands (optional, for tighter hand crop) ─────────────────────────
_mp_hands = None

def _try_init_mp_hands():
    global _mp_hands
    try:
        import mediapipe as mp
        _mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.4,
        )
        print("[GestureService] MediaPipe Hands ready (tight crop enabled)")
    except Exception as e:
        print(f"[GestureService] MediaPipe unavailable ({e}) — using center crop")


# ── Frame → PIL Image preprocessing ──────────────────────────────────────────

def _hand_roi(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    if _mp_hands is not None:
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = _mp_hands.process(rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                xs = [p.x for p in lm]
                ys = [p.y for p in lm]
                pad = 0.20
                x1 = max(0, int((min(xs) - pad) * w))
                y1 = max(0, int((min(ys) - pad) * h))
                x2 = min(w, int((max(xs) + pad) * w))
                y2 = min(h, int((max(ys) + pad) * h))
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    return crop
        except Exception:
            pass

    # Fallback: center square crop (60% of shorter side)
    side = int(min(h, w) * 0.6)
    cx, cy = w // 2, h // 2
    x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
    x2, y2 = min(w, x1 + side), min(h, y1 + side)
    return frame_bgr[y1:y2, x1:x2]


def _to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """Extract hand ROI from a BGR frame and return as PIL RGB Image."""
    roi = _hand_roi(frame_bgr)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ── Service class ─────────────────────────────────────────────────────────────

class GestureService:
    """
    ASL alphabet recognizer using HuggingFace SigLIP ViT.
    Same public interface as the previous ASL CNN service.
    """

    def __init__(self, groq_api_key: str = ""):
        from transformers import AutoImageProcessor, SiglipForImageClassification

        print(f"[GestureService] Loading {_MODEL_ID} (downloads on first run)…")
        self._processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
        self._model     = SiglipForImageClassification.from_pretrained(_MODEL_ID)
        self._model.eval()

        # Use the model's own label mapping (guaranteed correct order)
        self._id2label: dict[int, str] = {
            int(k): v for k, v in self._model.config.id2label.items()
        }
        num_classes = len(self._id2label)
        print(f"[GestureService] Ready — {num_classes} classes "
              f"({self._id2label[0]}–{self._id2label[num_classes-1]})")

        _try_init_mp_hands()

        self._lock = threading.Lock()

        self._groq_client = None
        if groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=groq_api_key)
                print("[GestureService] Groq sentence builder ready")
            except Exception as e:
                print(f"[GestureService] Groq unavailable: {e}")

    @property
    def t_max(self) -> int:
        return 20

    # ── Public API ─────────────────────────────────────────────────────────────

    def interpret_letter(self, jpeg_frames: list[bytes]) -> dict:
        letter, conf, alts = self._vote(jpeg_frames)
        if letter is None:
            return {"letter": "?", "confidence": 0.0,
                    "alternative": None, "note": "No hand detected"}
        return {
            "letter":      letter,
            "confidence":  round(conf, 3),
            "alternative": alts[0] if alts else None,
            "note":        None if conf >= 0.55 else "Low confidence — try again",
        }

    def interpret_word(self, jpeg_frames: list[bytes]) -> dict:
        letter, conf, alts = self._vote(jpeg_frames)
        if letter is None:
            return {
                "sign": "UNCLEAR", "english": "",
                "confidence": 0.0,
                "handshape_observed": "No hand detected",
                "alternatives": [], "note": "No hand detected",
            }
        return {
            "sign":               letter,
            "english":            f"Letter {letter}",
            "confidence":         round(conf, 3),
            "handshape_observed": f"ASL handshape for '{letter}'",
            "alternatives":       [f"Letter {a}" for a in alts],
            "note":               None if conf >= 0.55 else "Low confidence",
        }

    def build_sentence(self, signs: list[str]) -> dict:
        letters = [s[0].upper() for s in signs if s]
        word    = "".join(letters)

        if self._groq_client and len(letters) > 2:
            try:
                resp = self._groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content":
                         "You are an ASL fingerspelling interpreter. "
                         "Given a list of signed letters, produce the most likely "
                         "English word or phrase. "
                         'Respond ONLY with JSON: {"sentence":"...","confidence":0.8}'},
                        {"role": "user",
                         "content": f"Signed letters: {json.dumps(letters)}"},
                    ],
                    temperature=0.1, max_tokens=64,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                print(f"[GestureService] Groq error: {e}")

        return {"sentence": word, "confidence": 0.6,
                "note": f"Fingerspelled: {' '.join(letters)}"}

    def close(self):
        if _mp_hands is not None:
            try:
                _mp_hands.close()
            except Exception:
                pass

    # ── Internal ────────────────────────────────────────────────────────────────

    def _vote(self, jpeg_frames: list[bytes]) -> tuple[Optional[str], float, list[str]]:
        if not jpeg_frames:
            return None, 0.0, []

        votes: dict[str, list[float]] = {}

        with self._lock:
            for jpeg in jpeg_frames:
                try:
                    arr   = np.frombuffer(jpeg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    # Fix landscape rotation from Android front camera
                    h, w = frame.shape[:2]
                    if w > h:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    pil_img = _to_pil(frame)

                    inputs = self._processor(images=pil_img, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self._model(**inputs)
                        probs   = F.softmax(outputs.logits, dim=-1)[0]

                    idx    = int(probs.argmax())
                    conf   = float(probs[idx])
                    letter = self._id2label[idx]
                    votes.setdefault(letter, []).append(conf)

                except Exception as e:
                    print(f"[GestureService] frame error: {e}")

        if not votes:
            return None, 0.0, []

        best = max(votes, key=lambda k: sum(votes[k]) / len(votes[k]))
        conf = sum(votes[best]) / len(votes[best])
        alts = sorted(
            [k for k in votes if k != best],
            key=lambda k: -sum(votes[k]) / len(votes[k]),
        )[:2]

        print(f"[GestureService] vote: {best} ({conf*100:.0f}%)  "
              f"from {sum(len(v) for v in votes.values())}/{len(jpeg_frames)} frames")
        return best, conf, alts


# ── Singleton ──────────────────────────────────────────────────────────────────

_service: Optional[GestureService] = None


def init_gesture_service(groq_api_key: str = "") -> None:
    global _service
    if _service:
        return
    try:
        _service = GestureService(groq_api_key=groq_api_key)
    except Exception as e:
        print(f"[GestureService] Init FAILED: {e}")
        import traceback; traceback.print_exc()


def get_gesture_service() -> Optional[GestureService]:
    return _service
