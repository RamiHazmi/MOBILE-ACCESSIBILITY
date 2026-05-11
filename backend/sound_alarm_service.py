"""
backend/sound_alarm_service.py
════════════════════════════════
YAMNet-based environmental sound detection for deaf users.

Key fixes vs first version:
  • Confidence threshold lowered to 0.12 (phone mic → weaker signal)
  • Uses MAX score across time frames, not mean (short sounds get diluted)
  • Added broad YAMNet parent categories: Siren, Civil defense siren,
    Vehicle, Alarm, Smoke detector …
  • Prints top-5 scores every call so you can see what YAMNet detects
"""

import io
import csv
from typing import Optional

import numpy as np

# ─── Alert registry ─────────────────────────────────────────────────
# Key   = exact YAMNet display_name from yamnet_class_map.csv
# Value = (category, emoji, human description)
ALERT_LABELS: dict = {
    # ── Sirens / emergency ──────────────────────────────────────────
    "Siren":                            ("emergency", "🚨", "Siren detected"),
    "Civil defense siren":              ("emergency", "🚨", "Emergency siren"),
    "Ambulance (siren)":               ("emergency", "🚑", "Ambulance siren nearby"),
    "Emergency vehicle":               ("emergency", "🚨", "Emergency vehicle nearby"),
    "Police car (siren)":              ("emergency", "🚔", "Police siren nearby"),
    "Fire engine, fire truck (siren)": ("fire",      "🚒", "Fire truck siren"),

    # ── Fire / smoke ────────────────────────────────────────────────
    "Fire alarm":                      ("fire",      "🔥", "Fire alarm triggered"),
    "Smoke detector, smoke alarm":     ("fire",      "🔥", "Smoke alarm triggered"),
    "Alarm":                           ("alarm",     "🔔", "Alarm detected"),

    # ── Door ────────────────────────────────────────────────────────
    "Doorbell":                        ("door",      "🔔", "Someone at the door"),
    "Knock":                           ("door",      "🚪", "Knocking at the door"),
    "Door":                            ("door",      "🚪", "Door sound"),

    # ── Danger ──────────────────────────────────────────────────────
    "Screaming":                       ("danger",    "😱", "Screaming detected"),
    "Shout":                           ("danger",    "😱", "Shouting detected"),
    "Glass breaking":                  ("danger",    "🪟", "Glass breaking"),
    "Explosion":                       ("danger",    "💥", "Explosion detected"),
    "Gunshot, gunfire":                ("danger",    "⚠️",  "Gunshot detected"),
    "Bang":                            ("danger",    "💥", "Loud bang detected"),

    # ── Care ────────────────────────────────────────────────────────
    "Baby cry, infant cry":            ("care",      "👶", "Baby crying"),
    "Crying, sobbing":                 ("care",      "😢", "Someone crying"),

    # ── Call / alarm ────────────────────────────────────────────────
    "Telephone bell ringing":          ("call",      "📞", "Phone ringing"),
    "Ringtone":                        ("call",      "📞", "Ringtone"),
    "Alarm clock":                     ("alarm",     "⏰", "Alarm clock ringing"),
    "Beep, bleep":                     ("alarm",     "📳", "Alert beep"),

    # ── Vehicle ─────────────────────────────────────────────────────
    "Car alarm":                       ("vehicle",   "🚗", "Car alarm"),
    "Vehicle horn, car horn, honking": ("vehicle",   "📯", "Car horn"),
    "Bicycle bell":                    ("vehicle",   "🚲", "Bicycle bell"),

    # ── Animal ──────────────────────────────────────────────────────
    "Dog":                             ("animal",    "🐕", "Dog barking"),
    "Bark":                            ("animal",    "🐕", "Dog barking"),
}

# Lower threshold for phone mic audio (weaker signal than studio)
CONFIDENCE_THRESHOLD = 0.12

# ─── Model singleton ────────────────────────────────────────────────
_yamnet:      object       = None
_class_names: list         = []
_load_ok:     bool         = False


def load_yamnet() -> bool:
    """Lazy-load YAMNet from TF Hub. Thread-safe to call multiple times."""
    global _yamnet, _class_names, _load_ok
    if _yamnet is not None:
        return _load_ok
    try:
        import tensorflow as tf
        import tensorflow_hub as hub

        print("[alarm] Loading YAMNet from TF Hub…")
        _yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

        class_map_path = _yamnet.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(class_map_path) as resp:
            reader       = csv.DictReader(resp)
            _class_names = [row["display_name"] for row in reader]

        _load_ok = True
        print(f"[alarm] YAMNet ready — {len(_class_names)} classes")

        # Sanity-check: print which of our alert labels are found
        found   = [l for l in ALERT_LABELS if l in _class_names]
        missing = [l for l in ALERT_LABELS if l not in _class_names]
        print(f"[alarm] Alert labels found in model: {len(found)}/{len(ALERT_LABELS)}")
        if missing:
            print(f"[alarm] Missing labels (check spelling): {missing}")
        return True

    except Exception as exc:
        print(f"[alarm] YAMNet load FAILED: {exc}")
        import traceback; traceback.print_exc()
        return False


def is_ready() -> bool:
    return _load_ok


# ─── Analysis ───────────────────────────────────────────────────────

def analyze(pcm_bytes: bytes, sample_rate: int = 16000) -> list:
    """
    Analyse one chunk of PCM16 mono audio.

    Strategy: use MAX score across time frames rather than mean.
    Mean dilutes a short siren to near zero; max preserves the peak.

    Returns list of dicts (≤3, sorted by confidence desc):
        {label, emoji, category, description, confidence}
    """
    if not _load_ok:
        return []

    try:
        import tensorflow as tf

        # PCM16 → float32 in [-1, 1]
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # YAMNet needs at least ~0.48 s (one frame = 0.48 s)
        if len(pcm) < sample_rate // 2:
            print(f"[alarm] chunk too short: {len(pcm)} samples")
            return []

        waveform            = tf.constant(pcm, dtype=tf.float32)
        scores, _, _mel     = _yamnet(waveform)   # scores: (frames, 521)

        # Use MAX across time frames — catches short events like a single siren pulse
        max_scores  = tf.reduce_max(scores, axis=0).numpy()
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        # Blend: 60% max + 40% mean (responsive but not too noisy)
        combined    = 0.6 * max_scores + 0.4 * mean_scores

        # ── Debug: print top-5 only when not dominated by silence ──
        top5_idx = np.argsort(combined)[::-1][:5]
        top5     = [(
            _class_names[i] if i < len(_class_names) else f"#{i}",
            round(float(combined[i]), 3)
        ) for i in top5_idx]
        if top5[0][0] not in ("Silence", "White noise", "Pink noise") or top5[0][1] < 0.85:
            print(f"[alarm] top-5: {top5}")

        detections = []
        for label, (category, emoji, description) in ALERT_LABELS.items():
            if label not in _class_names:
                continue
            idx  = _class_names.index(label)
            conf = float(combined[idx])
            if conf >= CONFIDENCE_THRESHOLD:
                detections.append({
                    "label":       label,
                    "emoji":       emoji,
                    "category":    category,
                    "description": description,
                    "confidence":  round(conf, 3),
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        if detections:
            print(f"[alarm] 🔔 detections: {[(d['label'], d['confidence']) for d in detections]}")

        return detections[:3]

    except Exception as exc:
        print(f"[alarm] analyze error: {exc}")
        import traceback; traceback.print_exc()
        return []
